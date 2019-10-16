from Common import *

PhysicsObt = OrderedDict(
    ## Name : [number of object, scale]
    {
        "puppiJetEt"  : [5, 40],
        "puppiHT"     : [1, 40],
        "puppiMETEt"  : [1, 40],
        "puppiJetEta" : [5, 5],
        "puppiJetPhi" : [5, 5],
        "puppiMETPhi" : [1, 5],
        "EGEt" : [5, 20],
        "EGEta" : [5, 5],
        "EGPhi" : [5, 5],
        "pfMuonPt" : [2, 20],
        "pfMuonEta" : [2, 5],
        "pfMuonPhi" : [2, 5],
        "pfTauEt" : [5, 20],
        "pfTauEta" : [5, 5],
        "pfTauPhi" : [5, 5],
    }
)

modelname = "PuppiLep"

class autoencoder(nn.Module):
    def __init__(self,features):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(features, int(features*2/3)),#144,1280
            nn.ReLU(True),
            nn.Linear(int(features*2/3), int(features/3)), # 1280,64
            nn.ReLU(True),
            nn.Linear(int(features/3), int(features/3)), # 1280,64
            nn.ReLU(True),
            nn.Linear(int(features/3), int(features/3)), # 1280,64
            nn.ReLU(True),
            nn.Linear(int(features/3), int(features/3)), # 1280,64
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(int(features/3), int(features/3)), # 1280,64
            nn.ReLU(True),
            nn.Linear(int(features/3), int(features/3)), # 1280,64
            nn.ReLU(True),
            nn.Linear(int(features/3), int(features/3)),
            nn.ReLU(True),
            nn.Linear(int(features/3), int(features*2/3)),
            nn.ReLU(True),
            nn.Linear(int(features*2/3), features), 
            nn.ReLU(True))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
