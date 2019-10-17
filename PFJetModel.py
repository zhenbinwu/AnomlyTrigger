from Common import *

PhysicsObt = OrderedDict(
    ## Name : [number of object, scale]
    {
        "jetEt" :  [12, 40],
        "jetEta" : [12, 5],
        "jetPhi" : [12, 5],
    }
)

# globalcutstring = '(t["jetEt"] > 100).sum()>2'
modelname = "PFJet5"


class autoencoder(nn.Module):
    def __init__(self,features):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(features, int(features*2/3)),#144,1280
            nn.ReLU(True),
            nn.Linear(int(features*2/3), int(features/3)), # 1280,64
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(int(features/3), int(features*2/3)),
            nn.ReLU(True),
            nn.Linear(int(features*2/3), features), 
            nn.ReLU(True))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
