import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sim_data import generate_multimodal_data
from sklearn.decomposition import PCA
import seaborn as sns   
from evaluate_representations import plot_representations, evaluate_predictability


class MultimodalDataset(Dataset):
    """Dataset class for multimodal data."""
    def __init__(self, h1, h2, x1, x2, labels):
        self.h1 = torch.tensor(h1, dtype=torch.float32)
        self.h2 = torch.tensor(h2, dtype=torch.float32)
        self.x1 = torch.tensor(x1, dtype=torch.float32)
        self.x2 = torch.tensor(x2, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.h1[idx], self.h2[idx], self.x1[idx], self.x2[idx], self.labels[idx]


class Apollo(nn.Module):
    def __init__(self, encoders, decoders, n_train, z_sizes, shared_size,
                 modality_names, modality_shapes, ckpt_path):
        super(Apollo, self).__init__()
        self.z_sizes = z_sizes
        self.shared_size = shared_size
        self.n_train = n_train
        self.modality_names = modality_names
        self.modality_shapes = modality_shapes
        self.ckpt_path = ckpt_path

        # Replace nn.Parameter with nn.Embedding for modality-specific latents
        self.posterior_means = nn.ModuleDict({
            mod: nn.Embedding(n_train, z_sizes[mod])
            for mod in modality_names
        })

        # Shared latent embedding
        self.shared_post_mean = nn.Embedding(n_train, shared_size)

        # Encoders and decoders
        self.encoders = nn.ModuleDict(encoders)
        self.decoders = nn.ModuleDict(decoders)

    def get_latents(self):
        """Retrieve shared and modality-specific latent vectors for a batch."""
        z_shared = self.shared_post_mean
        z_mod = {
            mod: self.posterior_means[mod]
            for mod in self.modality_names
        }
        return z_shared, z_mod

    def parameters_for_optim(self):
        """Return all trainable parameters for the optimizer."""
        return (
            list(self.encoders.parameters()) +
            list(self.decoders.parameters()) +
            list(self.shared_post_mean.parameters()) +
            [p for mod in self.modality_names for p in self.posterior_means[mod].parameters()]
        )

    def load_from_checkpoint(self):
        ckpt_path = self.ckpt_path
        for modal_name in self.modality_names:
            self.encoders[modal_name].load_state_dict(torch.load(os.path.join(ckpt_path, f"{modal_name}_encoder.pth")))
            self.decoders[modal_name].load_state_dict(torch.load(os.path.join(ckpt_path, f"{modal_name}_decoder.pth")))

            z_mod = np.load(os.path.join(ckpt_path, f"modality_z_{modal_name}.npy"), allow_pickle=True)
            if isinstance(z_mod, np.ndarray) and z_mod.dtype == np.object_:
                z_mod = np.stack(z_mod)
            self.posterior_means[modal_name].weight.data.copy_(torch.from_numpy(z_mod.astype(np.float32)))

        z_shared = np.load(os.path.join(ckpt_path, 'shared_z.npy'), allow_pickle=True)
        if isinstance(z_shared, np.ndarray) and z_shared.dtype == np.object_:
            z_shared = np.stack(z_shared)
        self.shared_post_mean.weight.data.copy_(torch.from_numpy(z_shared.astype(np.float32)))


    def save_to_checkpoint(self):
        os.makedirs(self.ckpt_path, exist_ok=True)
        for modal_name in self.modality_names:
            torch.save(self.encoders[modal_name].state_dict(),
                    os.path.join(self.ckpt_path, f"{modal_name}_encoder.pth"))
            torch.save(self.decoders[modal_name].state_dict(),
                    os.path.join(self.ckpt_path, f"{modal_name}_decoder.pth"))
            # Save only the weights of the embeddings
            np.save(os.path.join(self.ckpt_path, f"modality_z_{modal_name}.npy"),
                    self.posterior_means[modal_name].weight.data.cpu().numpy())
        
        np.save(os.path.join(self.ckpt_path, 'shared_z.npy'),
                self.shared_post_mean.weight.data.cpu().numpy())


    def train(self, trainloader, dataset_name, lr_enc=0.001, lr_dec=0.001, lr_optim=0.001, epochs_enc=20, epochs_dec=20, shared_labels=None):
        decoder_params = [param for mod in self.modality_names for param in self.decoders[mod].parameters()]
        encoder_params = [param for mod in self.modality_names for param in self.encoders[mod].parameters()]
        posterior_means_params = list( self.posterior_means[self.modality_names[0]].parameters()) + list(self.posterior_means[self.modality_names[1]].parameters())
        shared_params = list(self.shared_post_mean.parameters())
        
        optimizer_dec = optim.Adam(decoder_params, lr=lr_dec)
        optimizer_latent = optim.Adam(posterior_means_params + shared_params, lr=lr_optim)
        optimizer_enc = optim.Adam(encoder_params, lr=lr_enc)
        # Save the model after training


        # Train decoders and latent representations
        dec_loss = self.optimize_latent(trainloader, lr_dec, epochs_dec, optimizer_dec, optimizer_latent, shared_labels)
        # Train encoders
        enc_loss = self.train_encoder(trainloader, lr_enc, epochs_enc, optimizer_enc)
        self.save_to_checkpoint()
        plt.figure(figsize=(12, 6))

        # Plot decoder loss
        plt.subplot(1, 2, 1)
        plt.plot(dec_loss, label='Decoder Loss', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Decoder Loss Trend')
        plt.grid(True)
        plt.legend()

        # Plot encoder loss
        plt.subplot(1, 2, 2)
        plt.plot(enc_loss, label='Encoder Loss', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Encoder Loss Trend')
        plt.grid(True)
        plt.legend()

        plt.savefig(os.path.join("./plots/%s/apollo_loss_trend.png" % dataset_name))
        plt.show()

        return dec_loss, enc_loss

    def optimize_latent(self, trainloader, lr, n_epochs, optimizer_dec, optimizer_latent, shared_labels):
        mse_loss_fn = nn.MSELoss()
        kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
        loss_trend = []
        
        for epoch in range(n_epochs):
            batch_ind = 0
            epoch_loss = []
            for data in trainloader:
                batch_loss = 0
                for m_ind, (mod, z) in enumerate(self.posterior_means.items()):
                    x_m = data[m_ind]
                    batch_size = x_m.size(0)
                    noise_m = torch.randn(batch_size, self.z_sizes[mod]) * 0.5
                    noise_s = torch.randn(batch_size, self.shared_size) * 0.5
                    batch_ids = torch.arange(batch_ind, batch_ind + batch_size, device=z.weight.device)
                    z_m = self.posterior_means[mod](batch_ids) + noise_m
                    z_s = self.shared_post_mean(batch_ids) + noise_s
                    reconst = self.decoders[mod](z_s, z_m)
                    
                    # Calculate MSE loss
                    mse_loss = mse_loss_fn(x_m, reconst)
                    
                    # Calculate KL divergence loss
                    actDecay = 1e-3
                    loss_lNorm_m = torch.mean(torch.linalg.norm(z_m, dim=1)) / z_m.size(1)
                    loss_lNorm_s = torch.mean(torch.linalg.norm(z_s, dim=1)) / z_s.size(1)
                    kl_loss = (loss_lNorm_m + loss_lNorm_s) * actDecay
                    # Total loss
                    total_loss = mse_loss + kl_loss
                    
                    optimizer_dec.zero_grad()
                    optimizer_latent.zero_grad()
                    total_loss.backward()
                    optimizer_dec.step()
                    optimizer_latent.step()
                    batch_loss += total_loss.item()
                batch_ind += batch_size     
                epoch_loss.append(batch_loss) 
            loss_trend.append(np.mean(epoch_loss))  
            # if epoch % 100 == 0:
            #     plot_representations((self.posterior_means["A"].weight.data.cpu().numpy(), self.posterior_means["B"].weight.data.cpu().numpy(), self.shared_post_mean.weight.data.cpu().numpy(), self.shared_post_mean.weight.data.cpu().numpy()), 
            #                 shared_labels, 'shared_apollo', dataset_name="simulated_apollo", modality_names=["A", "B"])    
        return loss_trend

    def train_encoder(self, trainloader, lr, n_epochs, optimizer):
        loss_trend = []
        for epoch in range(n_epochs):
            batch_ind = 0
            epoch_loss = []
            for batch in trainloader:
                batch_loss = 0
                for m_ind, (mod, encoder) in enumerate(self.encoders.items()):
                    x_m = batch[m_ind]
                    batch_size = x_m.size(0)
                    z_s, z_m = encoder(x_m)
                    batch_ids = torch.arange(batch_ind, batch_ind + batch_size, device=z_s.device)
                    shared_mse = (self.shared_post_mean(batch_ids) - z_s) ** 2
                    mod_mse = (self.posterior_means[mod](batch_ids) - z_m) ** 2
                    loss = shared_mse.mean() + mod_mse.mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                batch_loss += loss.item()
                batch_ind += batch_size 
                epoch_loss.append(batch_loss)
            loss_trend.append(np.mean(epoch_loss))  
        return loss_trend

    def merge_representations(self, x_batch):
        z_m_all, z_s_all = [], []
        for mod, encoder in self.encoders.items():
            z_s, z_m = encoder(x_batch[mod])
            z_m_all.append(z_m)
            z_s_all.append(z_s)
        z = torch.cat(z_m_all + [torch.mean(torch.stack(z_s_all), dim=0)], dim=-1)
        return z

    def generate_sample(self, x_batch, ind, ref_modality, target_modality, n_samples=1):
        x_m = x_batch[ref_modality][ind]
        z_ref, z_s, _ = self.encoders[ref_modality](x_m.unsqueeze(0))
        z_ref_dist = torch.cat([self.posterior_means[ref_modality], self.shared_post_mean], dim=-1)
        z_ref = torch.cat([z_ref, z_s], dim=-1)
        z_ref = nn.functional.normalize(z_ref, dim=1)
        z_ref_dist = nn.functional.normalize(z_ref_dist, dim=1)
        cosine_similarity = torch.mm(z_ref_dist, z_ref.t()).squeeze()
        top_k_indices = torch.topk(cosine_similarity, k=n_samples).indices
        similar_samples = self.posterior_means[target_modality][top_k_indices]
        z_s = z_s.repeat(len(similar_samples), 1)
        reconst_sample = self.decoders[target_modality](similar_samples, z_s, z_m, dim=-1)
        return reconst_sample

    def encode(self, x_batch):
        z_m_all, z_s_all = {}, {}
        for mod, encoder in self.encoders.items():
            x_m = x_batch[mod]
            z_s, z_m = encoder(x_m)
            z_s_all[mod] = z_s
            z_m_all[mod] = z_m
        return z_s_all, z_m_all

    def decode(self, z_s_all=None, z_m_all=None):
        x_recon = {}
        if z_s_all is None and z_m_all is None:
            for m_ind, (mod, decoder) in enumerate(self.decoders.items()):
                x_recon[mod] = decoder(torch.cat([self.posterior_means[mod][:5], self.shared_post_mean[:5]], dim=-1))
        else:
            for m_ind, (mod, decoder) in enumerate(self.decoders.items()):
                x_recon[mod] = decoder(torch.cat([z_m_all[mod], z_s_all[mod]], dim=-1))
        return x_recon

    def _frobenius_distance_cosine(self, X, Y):
        X_normalized = X / X.norm(dim=1, keepdim=True)
        Y_normalized = Y / Y.norm(dim=1, keepdim=True)
        cosine_similarity_matrix = torch.mm(X_normalized, Y_normalized.t())
        frobenius_norm = torch.norm(cosine_similarity_matrix, p='fro')
        return frobenius_norm


def main(dataset_name):
    """Main function to run the training pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate and load data
    if dataset_name == "simulated_apollo":
        loaded_data = np.load("./data/simulated_data_apollo.npz")
        input_dims = [80,40]
        shared_rank, specific_rank = 20, 20
    elif dataset_name == "simulated":
        loaded_data = np.load("./data/simplest_sim_nongaussian.npz")
        input_dims = [10,10]
        shared_rank, specific_rank = 5, 5
    h1, h2, x1, x2, labels = loaded_data["h1"], loaded_data["h2"], loaded_data["x1"], loaded_data["x2"], loaded_data["labels"]
    n_train = int(0.8*len(h1))
    n_val = int(0.1*len(h1))
    n_test = len(h1) - n_train - n_val
    # Create datasets
    dataset = MultimodalDataset(h1[:n_train], h2[:n_train], x1[:n_train], x2[:n_train], labels[:n_train])
    val_dataset = MultimodalDataset(h1[n_train:n_train+n_val], h2[n_train:n_train+n_val], x1[n_train:n_train+n_val], x2[n_train:n_train+n_val], labels[n_train:n_train+n_val])
    test_dataset = MultimodalDataset(h1[n_train+n_val:], h2[n_train+n_val:], x1[n_train+n_val:], x2[n_train+n_val:], labels[n_train+n_val:])
    
    # Create dataloaders
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=512, shuffle=True)

    class Encoder(nn.Module):
        def __init__(self, input_size, shared_size, modality_specific_size):
            super(Encoder, self).__init__()
            hidden_size = 20
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2_shared = nn.Linear(hidden_size, shared_size)
            self.fc2_modality_specific = nn.Linear(hidden_size, modality_specific_size)

        def forward(self, x):
            x = F.normalize(x, p=2, dim=1)
            x = F.relu(self.fc1(x))
            z_shared = self.fc2_shared(x)
            z_modality_specific = self.fc2_modality_specific(x)
            return z_shared, z_modality_specific

    class Decoder(nn.Module):
        def __init__(self, output_size, shared_size, modality_specific_size):
            super(Decoder, self).__init__()
            hidden_size = 20    
            self.fc1 = nn.Linear(shared_size + modality_specific_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)

        def forward(self, z_shared, z_modality_specific):
            z = torch.cat([z_shared, z_modality_specific], dim=-1)
            h = F.relu(self.fc1(z))
            h = self.fc2(h)
            return h

    if dataset_name == "simulated_apollo":
        prediction_labels = [labels[n_train+n_val:,0], labels[n_train+n_val:,1]]#, labels[n_train+n_val:,2]]
        # prediction_labels = [labels[:n_train,0], labels[:n_train,1]]#, labels[n_train+n_val:,2]]
        task_names = ['shared_apollo', 'A_apollo']#, 'B_apollo']
    elif dataset_name == "simulated":
        prediction_labels = [labels[n_train+n_val:,0], labels[n_train+n_val:,1], labels[n_train+n_val:,2]]
        task_names = ['Shared', 'A-specific', 'B-specific']
    # Initialize encoders for each modality
    test_batch = {"A": torch.Tensor(h1[n_train+n_val:]), "B": torch.Tensor(h2[n_train+n_val:])}
    z_sizes = {"A": specific_rank, "B": specific_rank}
    shared_size = shared_rank
    modality_names = ["A", "B"]
    h_sizes = {"A": input_dims[0], "B": input_dims[1]}
    encoders = {
        mod: Encoder(input_size=h_sizes[mod], shared_size=shared_size, modality_specific_size=z_sizes[mod])
        for mod in modality_names
    }
    decoders = {
        mod: Decoder(output_size=h_sizes[mod], shared_size=shared_size, modality_specific_size=z_sizes[mod])
        for mod in modality_names
    }
    model = Apollo(encoders, decoders, n_train=n_train, z_sizes=z_sizes, shared_size=shared_size, modality_names=modality_names, modality_shapes=h_sizes, ckpt_path="./checkpoints/apollo")
    dec_loss, enc_loss = model.train(dataloader, dataset_name=dataset_name, lr_enc=0.001, lr_dec=0.0001, lr_optim=0.001, epochs_enc=2000, epochs_dec=2000, shared_labels=prediction_labels[0])
    # model.load_from_checkpoint()
    z_s_all, z_m_all = model.encode(test_batch)
    # z_s_all, z_m_all = model.get_latents()
    # components = [
    #             ("Zs1", model.shared_post_mean.weight.data.cpu().numpy()),  # Shared representation from modality 1
    #             ("Zs2", model.shared_post_mean.weight.data.cpu().numpy()),  # Shared representation from modality 2
    #             ("Zm1", model.posterior_means["A"].weight.data.cpu().numpy()),  # Modality-specific representation from modality 1
    #             ("Zm2", model.posterior_means["B"].weight.data.cpu().numpy()),
    #         ] 

    components = [
            ("Zs1", z_s_all["A"].detach().cpu().numpy()),  # Shared representation from modality 1
            ("Zs2", z_s_all["B"].detach().cpu().numpy()),  # Shared representation from modality 2
            ("Zm1", z_m_all["A"].detach().cpu().numpy()),  # Modality-specific representation from modality 1
            ("Zm2", z_m_all["B"].detach().cpu().numpy()),
        ] 
    
    # z1 = torch.concat([z1m, z1s], dim=1)
    # z2 = torch.concat([z2m, z2s], dim=1)
    for task_ind, label_task in enumerate(prediction_labels):
        print(label_task.shape)
        label_task = label_task.squeeze()
        # plot_representations((model.posterior_means["A"].weight.data.cpu().numpy(), model.posterior_means["B"].weight.data.cpu().numpy(), model.shared_post_mean.weight.data.cpu().numpy(), model.shared_post_mean.weight.data.cpu().numpy()), 
        #                     label_task, task_names[task_ind], dataset_name="simulated_apollo", modality_names=["A", "B"])
        plot_representations((z_m_all["A"].detach().cpu().numpy(), z_s_all["A"].detach().cpu().numpy(), z_m_all["B"].detach().cpu().numpy(), z_s_all["B"].detach().cpu().numpy()), label_task, task_names[task_ind], dataset_name=dataset_name, modality_names=["A", "B"])
        result_dict = evaluate_predictability(components, label_task, task_names[task_ind], dataset_name)
        print(result_dict)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run multimodal projection training.')
    parser.add_argument('--dataset_name', type=str, choices=['simulated', 'simulated_apollo'], default='simulated',
                        help='Type of dataset to use: either "simulated" or "simulated_apollo".')
    args = parser.parse_args()
    dataset_name = args.dataset_name
    main(dataset_name)