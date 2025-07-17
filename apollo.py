import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
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
    """
    Simplified Multimodal representation fusion. This class integrates multiple
    modalities (e.g., images, text, audio, etc.) into a unified representation space.

    Attributes:
        encoders (dict): A dictionary of pretrained unimodal encoder models, one for each modality.
                         The key is the modality name identifier.
        decoders (dict): A dictionary of decoder models (not pretrained), one for each modality.
                         The key is the modality name identifier.
        train_ids (list): A list of identifiers for training data, used for indexing or batching.
        z_sizes (dict): Size of latent representation for each modality.
        shared_size (int): Size of the shared representation.
        modality_names (list): A list of names for each modality.
        modality_shapes (dict): The input shape for samples of each modality.
        ckpt_path (str): The path to the checkpoint directory where the model's trained parameters are stored or 
                         loaded from.
    """
    def __init__(self, encoders, decoders, n_train, z_sizes, shared_size,
                 modality_names, modality_shapes, ckpt_path):
        super(Apollo, self).__init__()
        self.z_sizes = z_sizes
        self.shared_size = shared_size
        self.n_train = n_train
        self.n_modalitites = len(modality_names)
        self.modality_names = modality_names
        self.modality_shapes = modality_shapes
        self.ckpt_path = ckpt_path
        # Initialize latent representations
        self.posterior_means = {}
        for mod, z_size in z_sizes.items():
            self.posterior_means[mod] = nn.Parameter(torch.randn(self.n_train, z_size))
        self.shared_post_mean = nn.Parameter(torch.randn(self.n_train, shared_size))
        # Initialize pretrained models
        self.encoders, self.decoders = {}, {}
        for model_name in modality_names:
            self.encoders[model_name] = encoders[model_name]
            self.decoders[model_name] = decoders[model_name]

    def load_from_checkpoint(self):
        ckpt_path = self.ckpt_path
        for modal_name in self.modality_names:
            self.encoders[modal_name].load_state_dict(torch.load(os.path.join(ckpt_path, f"{modal_name}_encoder.pth")))
            self.decoders[modal_name].load_state_dict(torch.load(os.path.join(ckpt_path, f"{modal_name}_decoder.pth")))
            self.posterior_means[modal_name] = torch.from_numpy(np.load(os.path.join(ckpt_path, f"modality_z_{modal_name}.npy")))
        self.shared_post_mean = torch.nn.Parameter(torch.from_numpy(np.load(os.path.join(ckpt_path, 'shared_z.npy'))))

    def save_to_checkpoint(self):
        os.makedirs(self.ckpt_path, exist_ok=True)
        for modal_name in self.modality_names:
            torch.save(self.encoders[modal_name].state_dict(), os.path.join(self.ckpt_path, f"{modal_name}_encoder.pth"))
            torch.save(self.decoders[modal_name].state_dict(), os.path.join(self.ckpt_path, f"{modal_name}_decoder.pth"))
            np.save(os.path.join(self.ckpt_path, f"modality_z_{modal_name}.npy"), self.posterior_means[modal_name].detach().cpu().numpy())
        np.save(os.path.join(self.ckpt_path, 'shared_z.npy'), self.shared_post_mean.detach().cpu().numpy())

    def train(self, trainloader, lr_enc=0.001, lr_dec=0.001, epochs_enc=200, epochs_dec=200):
        decoder_params = [param for mod in self.modality_names for param in self.decoders[mod].parameters()]
        encoder_params = [param for mod in self.modality_names for param in self.encoders[mod].parameters()]
        posterior_means_params = [self.posterior_means[mod] for mod in self.modality_names]
        
        optimizer_dec = optim.Adam(decoder_params + posterior_means_params, lr=lr_dec)
        optimizer_enc = optim.Adam(encoder_params, lr=lr_enc)
        # Save the model after training


        self.save_to_checkpoint()

        # Train decoders and latent representations
        dec_loss = self.optimize_latent(trainloader, lr_dec, epochs_dec, optimizer_dec)

        # Train encoders
        enc_loss = self.train_encoder(trainloader, lr_enc, epochs_enc, optimizer_enc)
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

        plt.savefig(os.path.join("./plots/apollo_loss_trend.png"))
        plt.show()

        return dec_loss, enc_loss

    def optimize_latent(self, trainloader, lr, n_epochs, optimizer):
        loss_fn = nn.MSELoss()
        loss_trend = []
        
        for epoch in range(n_epochs):
            batch_ind = 0
            epoch_loss = []
            for data in trainloader:
                batch_loss = 0
                for m_ind, (mod, z) in enumerate(self.posterior_means.items()):
                    x_m = data[m_ind]
                    batch_size = x_m.size(0)
                    noise_m = torch.randn(batch_size, self.z_sizes[mod]) * 0.1
                    noise_s = torch.randn(batch_size, self.shared_size) * 0.1
                    z_m = z[batch_ind:batch_ind + batch_size] + noise_m
                    z_s = self.shared_post_mean[batch_ind:batch_ind + batch_size] + noise_s
                    reconst = self.decoders[mod](z_s, z_m)
                    loss = loss_fn(x_m, reconst)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    batch_loss += loss.item()
                batch_ind += batch_size     
                epoch_loss.append(batch_loss) 
            loss_trend.append(np.mean(epoch_loss))      
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
                    shared_mse = (self.shared_post_mean[batch_ind:batch_ind + batch_size] - z_s) ** 2
                    mod_mse = (self.posterior_means[mod][batch_ind:batch_ind + batch_size] - z_m) ** 2
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


def main():
    """Main function to run the training pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from sim_data import generate_multimodal_data
    
    # Generate and load data
    try:
        loaded_data = np.load("./data/simplest_sim_nongaussian.npz")
    except FileNotFoundError:
        # Create data directory if it doesn't exist
        os.makedirs("./data", exist_ok=True)
        h1, h2, x1, x2, labels = generate_multimodal_data(
            n_samples=6000, mod_dim=10, save_path="./data/simulated_data.npz")
        loaded_data = np.load("./data/simulated_data.npz")
    h1, h2, x1, x2, labels = loaded_data["h1"], loaded_data["h2"], loaded_data["x1"], loaded_data["x2"], loaded_data["labels"]
    # Create datasets
    dataset = MultimodalDataset(h1[:4000], h2[:4000], x1[:4000], x2[:4000], labels[:4000])
    val_dataset = MultimodalDataset(h1[4000:5000], h2[4000:5000], x1[4000:5000], x2[4000:5000], labels[4000:5000])
    
    # Create dataloaders
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=512, shuffle=True)

    class Encoder(nn.Module):
        def __init__(self, input_size, shared_size, modality_specific_size):
            super(Encoder, self).__init__()
            self.fc1 = nn.Linear(input_size, input_size * 2)
            self.fc2_shared = nn.Linear(input_size * 2, shared_size)
            self.fc2_modality_specific = nn.Linear(input_size * 2, modality_specific_size)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            z_shared = self.fc2_shared(x)
            z_modality_specific = self.fc2_modality_specific(x)
            return z_shared, z_modality_specific

    class Decoder(nn.Module):
        def __init__(self, output_size, shared_size, modality_specific_size):
            super(Decoder, self).__init__()
            self.fc1 = nn.Linear(shared_size + modality_specific_size, output_size * 2)
            self.fc2 = nn.Linear(output_size * 2, output_size)

        def forward(self, z_shared, z_modality_specific):
            z = torch.cat([z_shared, z_modality_specific], dim=-1)
            h = F.relu(self.fc1(z))
            h = self.fc2(h)
            return h

    # Initialize encoders for each modality
    test_batch = {"A": torch.Tensor(h1[5000:]), "B": torch.Tensor(h2[5000:])}
    z_sizes = {"A": 5, "B": 5}
    shared_size = 10
    modality_names = ["A", "B"]
    h_sizes = {"A": 10, "B": 10}
    encoders = {
        mod: Encoder(input_size=h_sizes[mod], shared_size=shared_size, modality_specific_size=z_sizes[mod])
        for mod in modality_names
    }
    decoders = {
        mod: Decoder(output_size=h_sizes[mod], shared_size=shared_size, modality_specific_size=z_sizes[mod])
        for mod in modality_names
    }
    model = Apollo(encoders, decoders, n_train=4000, z_sizes=z_sizes, shared_size=shared_size, modality_names=modality_names, modality_shapes=h_sizes, ckpt_path="./checkpoints/apollo")
    dec_loss, enc_loss = model.train(dataloader, lr_enc=0.001, lr_dec=0.001, epochs_enc=200, epochs_dec=200)
    model.load_from_checkpoint()
    z_s_all, z_m_all = model.encode(test_batch)


    components = [
            ("Zs1", z_s_all["A"]),  # Shared representation from modality 1
            ("Zs2", z_s_all["B"]),  # Shared representation from modality 2
            ("Zm1", z_m_all["A"]),  # Modality-specific representation from modality 1
            ("Zm2", z_m_all["B"]),
        ] 
    prediction_labels = [labels[5000:,0], labels[5000:,1], labels[5000:,2]]
    task_names = ['shared', 'A', 'B']
    # z1 = torch.concat([z1m, z1s], dim=1)
    # z2 = torch.concat([z2m, z2s], dim=1)
    for name, z in components:
        print(name, z.shape)
    for task_ind, label_task in enumerate(prediction_labels):
        print(label_task.shape)
        label_task = label_task.squeeze()
        plot_representations((z_m_all["A"], z_m_all["B"], z_s_all["A"], z_s_all["B"]), label_task, task_names[task_ind], "apollo_simulation", modality_names=["A", "B"])
        evaluate_predictability(components, label_task, task_names[task_ind], "apollo_simulation")




    # # Assuming z_s_all and z_m_all are dictionaries with modality names as keys
    # # and tensors of shape (num_samples, feature_dim) as values
    # for label in range(3):  # Assuming there are 3 labels
    #     fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    #     fig.suptitle(f'PCA Projection for Label {label}')
    #     labels_test = labels[5000:, label]

    #     for i, (modality, z_s) in enumerate(z_s_all.items()):
    #         # Filter based on label

    #         # Perform PCA
    #         pca_s = PCA(n_components=2)
    #         pca_m = PCA(n_components=2)
    #         z_s_pca = pca_s.fit_transform(z_s_all[modality].detach().cpu().numpy())
    #         z_m_pca = pca_m.fit_transform(z_m_all[modality].detach().cpu().numpy())

    #         # Plot shared component
    #         axes[i, 0].scatter(z_s_pca[:, 0], z_s_pca[:, 1], c=labels_test)
    #         axes[i, 0].set_title('Shared Component')
    #         axes[i, 0].set_xlabel('PC1')
    #         axes[i, 0].set_ylabel('PC2')

    #         # Plot modality-specific component
    #         axes[i, 1].scatter(z_m_pca[:, 0], z_m_pca[:, 1], c=labels_test)
    #         axes[i, 1].set_title('Modality-Specific Component')
    #         axes[i, 1].set_xlabel('PC1')
    #         axes[i, 1].set_ylabel('PC2')

    #     # for ax in axes:
    #     #     ax.legend()
    #     #     ax.grid(True)

    #     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    #     plt.savefig(os.path.join(f"./plots/apollo_pca_projection_label_{label}.png"))
    

if __name__ == "__main__":
    main()