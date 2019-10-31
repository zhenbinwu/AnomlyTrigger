from Common import *

PhysicsObt = OrderedDict(
    ## Name : [number of object, scale]
    {
        "puppiHT"     : [1, 40],
        "puppiMETEt"  : [1, 40],
        "puppiMETPhi" : [1, 5],
        "nPuppiJets"  : [1, 0],
        "puppiJetEt"  : [5, 40],
        "puppiJetEta" : [5, 5],
        "puppiJetPhi" : [5, 5],
    }
)

# globalcutfunc = lambda t : (t["puppiJetEt"] > 10).sum()>2

modelname = "PuppiJetMETHT_NJet_Deep_Reorg"

class autoencoder(nn.Module):
    def __init__(self,features):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(features, int(features*2/3)),#144,1280
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(int(features*2/3), int(features/3)), # 1280,64
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(int(features/3), int(features/3)), # 1280,64
            nn.ReLU(True),
            nn.Dropout(0.2),
			nn.Linear(int(features/3), int(features/3)), # 1280,64
			nn.ReLU(True),
            nn.Dropout(0.2),
			nn.Linear(int(features/3), int(features/3)), # 1280,64
			nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
			nn.Linear(int(features/3), int(features/3)), # 1280,64
			nn.ReLU(True),
            nn.Dropout(0.2),
			nn.Linear(int(features/3), int(features/3)), # 1280,64
			nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(int(features/3), int(features/3)),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(int(features/3), int(features*2/3)),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(int(features*2/3), features), 
            nn.ReLU(True))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
