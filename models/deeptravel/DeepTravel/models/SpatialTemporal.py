import torch
import torch.nn as nn


class SpatialTemporal(nn.Module):
    """Generates feature vectors Vsp and Vtp for each particular cell"""

    spatial_emb_dims = [
        ('G_X', 256, 100),
        ('G_Y', 256, 100)
    ]

    temporal_emb_dims = [
        ('day_bin', 7, 100),
        ('hour_bin', 24, 100),
        ('time_bin', 288, 100)
    ]

    def __init__(self):
        super(SpatialTemporal, self).__init__()
        self.build()

    def build(self):
        for name, dim_in, dim_out in (SpatialTemporal.spatial_emb_dims + SpatialTemporal.temporal_emb_dims):
            self.add_module(name + '_em', nn.Embedding(dim_in, dim_out))

        for module in self.modules():
            if type(module) is not nn.Embedding:
                continue
            nn.init.uniform_(module.state_dict()['weight'], a=-1, b=1)

    def forward(self, stats, temporal, spatial):

        V_tp = []
        for name, dim_in, dim_out in SpatialTemporal.temporal_emb_dims:
            embed = getattr(self, name + '_em')
            temporal_t = temporal[name].view(-1, 1)
            temporal_t = torch.squeeze(embed(temporal_t))
            # print(temporal_t.shape)
            V_tp.append(temporal_t.reshape(-1,100))

        V_sp = []
        for name, dim_in, dim_out in SpatialTemporal.spatial_emb_dims:
            embed = getattr(self, name + '_em')
            spatial_t = spatial[name].view(-1, 1)
            spatial_t = torch.squeeze(embed(spatial_t))
            V_sp.append(spatial_t.reshape(-1, 100))
        try:
            V_tp = torch.cat(V_tp, dim=1)           # [300]
        except Exception as e:
            print(V_tp)
            raise e
        # print(V_tp.shape)
        V_sp = torch.cat(V_sp, dim=1)           # [200]
        return V_sp, V_tp
