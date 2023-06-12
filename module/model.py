import torch
import torch.nn as nn


from einops import repeat, rearrange

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block


class ViTEncoderProjHead(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=12,
                 num_head=3,
                 
                 out_ndim=128,
                 ) -> None:
        super().__init__()
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))

        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))

        self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        feat_dim = self.pos_embedding.shape[-1] 

        self.feat_dim = feat_dim

        self.fc = nn.Sequential( 
            nn.Linear(feat_dim, feat_dim), 
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(), 
            nn.Linear(feat_dim, feat_dim), 
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(), 
            nn.Linear(feat_dim, out_ndim), 
        )
        
        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img, index):
        patches = self.patchify(img) # patch embedding, shape: (batch_size, emb_dim, ngrid, ngrid)
        patches = rearrange(patches, 'b c h w -> (h w) b c') # (npatch, batch_size, emb_dim)
        patches = patches + self.pos_embedding # (npatch, batch_size, emb_dim)
        index = repeat(index, 't b -> t b c', c=patches.shape[-1])
        patches = torch.gather(patches, 0, index)  # shape: (npatch_visible, batch_size, emb_dim)
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0) # (1 + npatch_visible, batch_size, emb_dim)
        patches = rearrange(patches, 't b c -> b t c') # (batch_size, 1 + npatch_visible, emb_dim)
        features = self.layer_norm(self.transformer(patches)) # (batch_size, 1 + npatch_visible, emb_dim)
        features = rearrange(features, 'b t c -> t b c') # (1 + npatch_visible, batch_size, emb_dim)
        out = self.fc(features[0]) # (batch_size, out_dim)
        return out


class ViTEncoderPredHead(torch.nn.Module):
    def __init__(self, dim=128, mlp_dim=512, T=1.0,image_size=32, patch_size=2, emb_dim=192, num_layer=12, num_head=3):
        super().__init__()
        self.base_encoder = ViTEncoderProjHead(image_size=image_size, patch_size=patch_size, emb_dim=emb_dim, num_layer=num_layer, num_head=num_head)
        self.T = T
        self.predictor = self._build_mlp(3, dim, mlp_dim ,dim)
        hidden_dim = self.base_encoder.feat_dim
        self.base_encoder.fc = self._build_mlp(3, hidden_dim, mlp_dim ,dim)

    def forward(self, x1, x2, idxs1, idxs2):
        z1 = self.base_encoder(x1, idxs1)
        z2 = self.base_encoder(x2, idxs2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        
        return p1, p2, z1, z2

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)


class Vit(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=12,
                 num_head=3,
                 out_ndim=128,
                 ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))

        self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.fc = torch.nn.Linear(self.pos_embedding.shape[-1], out_ndim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embedding', 'cls_token', 'dist_token'}
        
    def forward(self, img, out_feat=False):
        patches = self.patchify(img) # patch embedding, shape: (batch_size, emb_dim, ngrid, ngrid)
        patches = rearrange(patches, 'b c h w -> (h w) b c') # (npatch, batch_size, emb_dim)
        patches = patches + self.pos_embedding # (npatch, batch_size, emb_dim)

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0) 
        patches = rearrange(patches, 't b c -> b t c') 
        features = self.layer_norm(self.transformer(patches)) 
        features = rearrange(features, 'b t c -> t b c') 

        if out_feat:
            return features[0]

        out = self.fc(features[0]) # (batch_size, out_dim)

        return out



