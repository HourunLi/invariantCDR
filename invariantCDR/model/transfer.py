import torch
import torch.nn as nn

class FactorDomainTransformer(nn.Module):
    def __init__(self, args):
        super(FactorDomainTransformer, self).__init__()
        self.K = args.num_latent_factors 
        self.d = args.feature_dim // self.K
        self.shared_user = args.shared_user
        self.transformers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d, self.d),
                nn.LeakyReLU(args.leakey),
                nn.Linear(self.d, self.d),
            )
            for _ in range(args.num_latent_factors)
        ])

    def forward(self, user_embeddings, sim_matrix, item_embeddings, UV):
        transformed_embeddings = []
        batch_size = user_embeddings.size()[0]
        UV_dense = UV.to_dense()
        UV_shared = UV_dense[:self.shared_user]
        for i, transformer in enumerate(self.transformers):
            factor_user_embedding = user_embeddings[:, i, :]
            factor_item_embedding = item_embeddings[:, i, :]
            # print(factor_user_embedding.size(), factor_item_embedding.size())
            factor_sim_matrix = sim_matrix[i]
            transformed_embedding = transformer(factor_user_embedding)
            # print(transformed_embedding.size())
            
            factor_sim_matrix_shared = factor_sim_matrix[:batch_size, :self.shared_user]
            weighted_interactions = torch.mm(factor_sim_matrix_shared, UV_shared)
            weighted_item_embeddings = torch.mm(weighted_interactions, factor_item_embedding)
            # print(weighted_item_embeddings.size())
            emb = (transformed_embedding + weighted_item_embeddings)/2
            transformed_embeddings.append(emb.unsqueeze(1))
        
        transformed_embeddings = torch.cat(transformed_embeddings, dim=1)
        return transformed_embeddings
        
    def forward_user(self, embeddings):
        transformed_embeddings = []
        for i, transformer in enumerate(self.transformers):
            factor_embedding = embeddings[:, i, :]
            transformed_embedding = transformer(factor_embedding)
            transformed_embeddings.append(transformed_embedding.unsqueeze(1))
            
        transformed_embeddings = torch.cat(transformed_embeddings, dim=1)
        # print("transformed_embeddings size: {}".format(transformed_embeddings.size()))
        return transformed_embeddings
