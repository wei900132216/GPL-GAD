import argparse
import random
import torch
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.data.utils import load_graphs, load_info
from ogb.nodeproppred  import DglNodePropPredDataset
from sklearn.model_selection import train_test_split
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from get_args import get_my_args

from dgl.data import FraudAmazonDataset, FraudYelpDataset


def set_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
def node_mask(train_mask,mask_rate):
    mask_rate=int(mask_rate*10)
    count=0
    for i in range(train_mask.shape[0]):
        if train_mask[i]==True:
            count=count+1
            if count<=mask_rate:
                train_mask[i]=False
                count=count+1
            if count==10:
                count=0
    return train_mask
def gen_mask(g, train_rate, val_rate, IR, IR_set):
    labels = g.ndata['label']
    g.ndata['label'] = labels.long()
    labels = np.array(labels)
    n_nodes = len(labels)
    if IR_set == 0:
        index = list(range(n_nodes))
    # Unbalanced sampling based on IR
    else:
        fraud_index = np.where(labels == 1)[0].tolist()
        benign_index = np.where(labels == 0)[0].tolist()
        if len(np.unique(labels)) == 3:
            Courier_index = np.where(labels == 2)[0].tolist()
        if IR < (len(fraud_index) / len(benign_index)):
            number_sample = int(IR * len(benign_index))
            sampled_fraud_index = random.sample(fraud_index, number_sample)
            sampled_benign_index = benign_index
            if len(np.unique(labels)) == 3:
                sampled_Courier_index = random.sample(Courier_index, number_sample)
        else:
            number_sample = int(len(fraud_index) / IR)
            sampled_benign_index = random.sample(benign_index, number_sample)
            sampled_fraud_index = fraud_index
            if len(np.unique(labels)) == 3:
                sampled_Courier_index = Courier_index
        if len(np.unique(labels)) == 2:
            index = sampled_benign_index + sampled_fraud_index
        else:
            index = sampled_benign_index + sampled_fraud_index + sampled_Courier_index
        labels = labels[index]

    train_idx, val_test_idx, _, y_validate_test = train_test_split(index, labels, stratify=labels,
                                                                   train_size=train_rate, test_size=1 - train_rate,
                                                                   random_state=2, shuffle=True)
    val_idx, test_idx, _, _ = train_test_split(val_test_idx, y_validate_test, train_size=val_rate / (1 - train_rate),
                                               test_size=1 - val_rate / (1 - train_rate),
                                               random_state=2, shuffle=True)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    g.ndata['train_mask'] = train_mask
    g.ndata['val_mask'] = val_mask
    g.ndata['test_mask'] = test_mask
    return g, train_idx

class Dataset:
    def __init__(self, name='tfinance', homo=True, add_self_loop=True, to_bidirectional=False, to_simple=True):
        if name == 'yelp':
            dataset = FraudYelpDataset()
            graph = dataset[0]
            graph.ndata['train_mask'] = graph.ndata['train_mask'].bool()
            graph.ndata['val_mask'] = graph.ndata['val_mask'].bool()
            graph.ndata['test_mask'] = graph.ndata['test_mask'].bool()
            if homo:
                graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])

        elif name == 'amazon':
            dataset = FraudAmazonDataset()
            graph = dataset[0]
            graph.ndata['train_mask'] = graph.ndata['train_mask'].bool()
            graph.ndata['val_mask'] = graph.ndata['val_mask'].bool()
            graph.ndata['test_mask'] = graph.ndata['test_mask'].bool()
            graph.ndata['mark'] = graph.ndata['train_mask']+graph.ndata['val_mask']+graph.ndata['test_mask']
            if homo:
                graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask', 'mark'])

        else:
            graph = dgl.load_graphs('./data/'+name)[0][0]
        graph.ndata['feature'] = graph.ndata['feature'].float()
        graph.ndata['label'] = torch.argmax(graph.ndata['label'].long(), dim=1)

        self.name = name
        self.graph = graph
        if add_self_loop:
            self.graph = dgl.add_self_loop(self.graph)
        if to_bidirectional:
            self.graph = dgl.to_bidirected(self.graph, copy_ndata=True)
        if to_simple:
            self.graph = dgl.to_simple(self.graph)

    def split(self, samples=20):
        labels = self.graph.ndata['label']
        n = self.graph.num_nodes()

        if 'mark' in self.graph.ndata:
            index = self.graph.ndata['mark'].nonzero()[:, 0].numpy().tolist()
        else:
            index = list(range(n))

        train_mask = torch.zeros(n).bool()
        val_mask = torch.zeros(n).bool()
        test_mask = torch.zeros(n).bool()

        if self.name in ['tolokers', 'questions']:
            train_ratio, val_ratio = 0.5, 0.25
        elif self.name in ['tsocial', 'tfinance', 'reddit', 'weibo']:
            train_ratio, val_ratio = 0.5, 0.1
        if self.name in ['amazon', 'yelp', 'elliptic', 'dgraphfin']:
            train_mask = self.graph.ndata['train_mask']
            val_mask = self.graph.ndata['val_mask']
            test_mask = self.graph.ndata['test_mask']
        else:
            seed = 3407
            set_seed(seed)
            idx_train, idx_rest, y_train, y_rest = train_test_split(
                index, labels[index], stratify=labels[index], train_size=train_ratio, random_state=seed, shuffle=True
            )
            idx_valid, idx_test, y_valid, y_test = train_test_split(
                idx_rest, y_rest, stratify=y_rest, train_size=int(len(index) * val_ratio), random_state=seed,
                shuffle=True
            )
            train_mask[idx_train] = 1
            val_mask[idx_valid] = 1
            test_mask[idx_test] = 1

        # pos_index = np.where(labels == 1)[0]
        # neg_index = list(set(index) - set(pos_index))
        # pos_train_idx = np.random.choice(pos_index, size=2 * samples, replace=False)
        # neg_train_idx = np.random.choice(neg_index, size=8 * samples, replace=False)
        # train_idx = np.concatenate([pos_train_idx[:samples], neg_train_idx[:4 * samples]])
        # train_mask[train_idx] = 1
        # val_idx = np.concatenate([pos_train_idx[samples:], neg_train_idx[4 * samples:]])
        # val_mask[val_idx] = 1
        # test_mask[index] = 1
        # test_mask[train_idx] = 0
        # test_mask[val_idx] = 0
        print(torch.sum(test_mask).item())
        print(torch.sum(val_mask).item())
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask
def my_load_data(args):
    if args.dataset=='cora' or args.dataset=='citeseer' or args.dataset=='pubmed' or args.dataset=='reddit':
        data = load_data(args)
        g = data[0]
        features = g.ndata['feat']
        labels = g.ndata['label']
        train_mask = g.ndata['train_mask']
        if args.half:
            train_mask=node_mask(train_mask,args.mask_rate)
        else:
            train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        in_feats = features.shape[1]
        n_classes = data.num_classes
        n_edges = g.num_edges()
    elif args.dataset=='Fraud_yelp' or args.dataset=='Fraud_amazon':
        if args.dataset=='Fraud_yelp':
            data = dgl.data.FraudDataset('yelp')
        else:
            data = dgl.data.FraudDataset('amazon')
        g = data[0]
        g=dgl.to_homogeneous(g,ndata=['feature','label','train_mask','val_mask','test_mask'])
        features = g.ndata['feature'].to(torch.float32)
        g.ndata['feat'] = g.ndata.pop('feature')
        labels = g.ndata['label']
        train_mask = g.ndata['train_mask']
        if args.half:
            train_mask=node_mask(train_mask,args.mask_rate)
        else:
            train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        in_feats = features.shape[1]
        n_classes = data.num_classes
        n_edges = data.graph.number_of_edges()
    elif args.dataset=='Sichuan' or args.dataset=='BUPT':
        if args.dataset=='Sichuan':
            dataset, _ = load_graphs("./data/Sichuan_tele.bin")  # glist will be [g1]
            n_classes = load_info("./data/Sichuan_tele.pkl")['num_classes']
        else:
            dataset, _ = load_graphs("./data/BUPT_tele.bin")  # glist will be [g1]
            n_classes = load_info("./data/BUPT_tele.pkl")['num_classes']
        # {0: 4144, 1: 1962}
        g,_ = gen_mask(dataset[0], 0.2, 0.2, 0.1, 0)
        # for e in g.etypes:
        #     g = dgl.remove_self_loop(g, etype=e)
        #     g = dgl.add_self_loop(g, etype=e)
        features = g.ndata['feat'].float()
        labels = g.ndata['label']
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        in_feats = features.shape[1]
        n_edges = g.num_edges()

    elif args.dataset == 'tsocial' or 'tfinance':
        data = Dataset(name='tfinance')
        data.split()
        g = data.graph
        # g = dgl.to_homogeneous(g, ['feature', 'label', 'train_masks', 'val_masks', 'test_masks'])
        features = g.ndata['feature'].float()
        g.ndata['feat'] = g.ndata.pop('feature')
        labels = g.ndata['label']
        train_mask = g.ndata['train_mask']
        print(torch.sum(g.ndata['train_mask']).item())
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        print(torch.sum(g.ndata['test_mask']).item())
        in_feats = features.shape[1]
        n_edges = g.num_edges()
        n_classes = len(torch.unique(labels))
    elif args.dataset=='elliptic':
        g = dgl.load_graphs('data/elliptic')[0][0]
        g = dgl.to_homogeneous(g,ndata=['feature','label', 'train_mask', 'val_mask', 'test_mask'])
        features = g.ndata['feature'].float()
        g.ndata['feat'] = g.ndata.pop('feature')
        labels = g.ndata['label']
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        in_feats = features.shape[1]
        n_edges = g.num_edges()
        n_classes = len(torch.unique(labels))
    elif args.dataset=='CoraFull':
        data = dgl.data.CoraFullDataset()
        g = data[0]
        features = g.ndata['feat']
        labels = g.ndata['label']
        ind=torch.Tensor(random.choices([0,1,2],weights=[0.3,0.1,0.6],k=features.shape[0]))
        g.ndata['train_mask']= (ind==0)
        g.ndata['val_mask']= (ind==1)
        g.ndata['test_mask']= (ind==2)
        train_mask = g.ndata['train_mask']
        if args.half:
            train_mask=node_mask(train_mask,args.mask_rate)
        else:
            train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        in_feats = features.shape[1]
        n_classes = data.num_classes
        n_edges = g.number_of_edges()
    elif args.dataset=='AmazonCoBuyComputer' or args.dataset=='AmazonCoBuyPhoto' or args.dataset=='CoauthorCS' :
        if args.dataset=='AmazonCoBuyComputer':
            data = dgl.data.AmazonCoBuyComputerDataset()
        elif args.dataset=='AmazonCoBuyPhoto':
            data = dgl.data.AmazonCoBuyPhotoDataset()
        elif args.dataset=='CoauthorCS':
            data = dgl.data.CoauthorCSDataset()
        g = data[0]
        features = g.ndata['feat']  # get node feature
        labels = g.ndata['label']  # get node labels
        ind=torch.Tensor(random.choices([0,1,2],weights=[0.1,0.3,0.6],k=features.shape[0]))
        g.ndata['train_mask']= (ind==0)
        g.ndata['val_mask']= (ind==1)
        g.ndata['test_mask']= (ind==2)
        train_mask = g.ndata['train_mask']
        if args.half:
            train_mask=node_mask(train_mask,args.mask_rate)
        else:
            train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        in_feats = features.shape[1]
        n_classes = data.num_classes
        n_edges = g.number_of_edges()
    elif args.dataset=='ogbn-arxiv':
        dataset = DglNodePropPredDataset(name='ogbn-arxiv')

        split_idx = dataset.get_idx_split()
        g, labels = dataset[0]
        g = dgl.add_reverse_edges(g)
        features = g.ndata['feat']
        g.ndata['label']=labels.view(-1,)
        ind=torch.zeros(labels.shape,dtype=bool)
        ind[split_idx['train']]=True
        g.ndata['train_mask']= ind.view(-1,)
        ind=torch.zeros(labels.shape,dtype=bool)
        ind[split_idx['valid']]=True
        g.ndata['val_mask']= ind.view(-1,)
        ind=torch.zeros(labels.shape,dtype=bool)
        ind[split_idx['test']]=True
        g.ndata['test_mask']= ind.view(-1,)
        train_mask = g.ndata['train_mask']
        if args.half:
            train_mask=node_mask(train_mask,args.mask_rate)
        else:
            train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        in_feats = features.shape[1]
        n_classes = dataset.num_classes
        n_edges = g.number_of_edges()
        labels=labels.view(-1,)
        pass
    else:
        g=None
        features=None
        labels=None
        train_mask=None
        val_mask=None
        test_mask=None
        in_feats=None
        n_classes=None
        n_edges=None
    return g,features,labels,train_mask,val_mask,test_mask,in_feats,n_classes,n_edges

def evaluate(model, graph, nid, batch_size, device,sample_list):
    sampler = dgl.dataloading.MultiLayerNeighborSampler(sample_list)
    valid_dataloader = dgl.dataloading.DataLoader(graph, nid.int(), sampler,batch_size=batch_size,shuffle=False,drop_last=False,device=device)

    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for input_nodes, output_nodes, mfgs in valid_dataloader:
            inputs = mfgs[0].srcdata['feat']
            labels.append(mfgs[-1].dstdata['label'].numpy(force=True))
            predictions.append(model(mfgs, inputs).argmax(1).numpy(force=True))
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        recall = recall_score(labels, predictions, average='weighted')
        conf_matrix = confusion_matrix(labels, predictions)
    return accuracy, f1, recall, conf_matrix

def constraint(device,prompt):
    if isinstance(prompt,list):
        sum=0
        for p in prompt:
            sum=sum+torch.norm(torch.mm(p,p.T)-torch.eye(p.shape[0]).to(device))
        return sum/len(prompt)
    else:
        return torch.norm(torch.mm(prompt,prompt.T)-torch.eye(prompt.shape[0]).to(device))

def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def get_init_info(args):
    g,features,labels,train_mask,val_mask,test_mask,in_feats,n_classes,n_edges=my_load_data(args)
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.int().sum().item(),
           val_mask.int().sum().item(),
           test_mask.int().sum().item()))

    if args.gpu < 0:
        device='cpu'
    else:
        device='cuda:'+str(args.gpu)
        torch.cuda.set_device(args.gpu)

    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
    print("use cuda:", args.gpu)

    train_nid = train_mask.nonzero().squeeze()
    val_nid = val_mask.nonzero().squeeze()
    test_nid = test_mask.nonzero().squeeze()
    g = dgl.remove_self_loop(g)
    n_edges = g.number_of_edges()
    if args.gpu >= 0:
        g = g.int().to(args.gpu)
    return g,features,labels,in_feats,n_classes,n_edges,train_nid,val_nid,test_nid,device
if __name__ == '__main__':
    args = get_my_args()
    print(args)
    info=my_load_data(args)
    g=info[0]
    labels=torch.sparse.sum(g.adj(),1).to_dense().int().view(-1,)
    print(labels)
    li=list(set(labels.numpy()))
    for i in range(labels.shape[0]):
        labels[i]=li.index(labels[i])
    print(set(labels.numpy()))
    #print(info)