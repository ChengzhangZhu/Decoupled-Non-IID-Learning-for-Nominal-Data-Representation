"""
This code provides a demo how to use nonBEND class for non-IID nominal data representation
"""

from nonBEND import  nonBEND
import pickle
import argparse

parser = argparse.ArgumentParser(description='NonBEND Representation')
parser.add_argument('--epochs', default=30, help='Set the epochs for Gibbs sampling and variational inference',
                    dest='epochs', type=int)
parser.add_argument('--data_set', default=None, help='Specific the data set', dest='data_set', type=str)
parser.add_argument('--burnin', default=10000, help='The burn in steps for Gibbs sampling', dest='burnin', type=int)
parser.add_argument('--verbose', default='n', help='Print detail information during sampling', dest='verbose', type=str)
parser.add_argument('--T', default='30', help='The trunked size for variational inference', dest='T', type=int)
parser.add_argument('--infer', default='vi', help='variational inference (vi) or Gibbs sampling (gbs)', dest='infer', type=str)
args = parser.parse_args()

if args.verbose == 'n' or args.verbose == 'N':
    verbose = False
else:
    verbose = True

data_package = pickle.load(open('./Data/'+args.data_set+'.pkl', 'rb'))  # load data and label
data = data_package['data']
label = data_package['label']
model = nonBEND(prt = verbose, name = args.data_set, maxEpoch = args.epochs, burnin = args.burnin)  # model initialization
if args.infer == 'vi':
    model.fit_vi(data, T=args.T, n_iter=args.epochs)
else:
    model.fit_gbs(data)
pickle.dump(model, open('./Model/'+args.data_set+'.model', 'wb'))
pickle.dump((model.embedding, label), open('./Representation/'+args.data_set+'.embedding', 'wb'))