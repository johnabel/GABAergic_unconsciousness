"""
OOP code for consciousness classification.

JHA

sklearn needs to be version 0.21.2
"""
import os
import warnings
import pickle
from time import time
from itertools import combinations
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn import decomposition, discriminant_analysis, svm, linear_model
import matplotlib.pyplot as plt
import pandas as pd
from hmmlearn import hmm as hmm_model

# ignore FutureWarning errors which clutter the outputs
warnings.simplefilter(action='ignore', category=FutureWarning)

##########################
###### GLOBAL VARS #######
##########################
# CNN features are the only ones not being generated online so we need file paths to where its stored on disk
FP_BTLNCKS_VOLUNTEER = 'Data/Volunteer_CNN/btlnck_df.csv'

n_pcs_cnn = 10 #number of principal components for PCA of CNN bottlenecks

if not os.path.exists(FP_BTLNCKS_VOLUNTEER):
    warnings.warn(f"filepath to CNN bottlenecks for volunteer data doesn't exist\nFP passed: {FP_BTLNCKS_VOLUNTEER}")

class ConsciousnessClassifier(object):
    """Consciousness classifier object that can use a variety of algorithms 
    to calculate the probability of consciousness from a spectogram of Fp1.
    Has been tested on Volunteer data from Purdon et al 2013 and on OR data.
    """

    def __init__(self, name=None):
        """
        An object for writing, fitting, and assessing models
        """
        self.name = name
        self.datasets = {}
        self.models = {} # option for multiple models

    def load_dataset(self, path, names, dsetname, subjects='OR'):
        """loads dataset as a dictionary and gets added to the self.datasets attribute
        NOTE: This object is built to have separate data sets for train and test (internal validation or external validation)
        
        Parameters
        ----------
        path : str
            directory path to folder containing by case preprocessed data
            e.g.~/Dropbox (Partners HealthCare)/HumanSignalsData/projects/consciousness_classifier/volunteer/by_case/
        names : list(str)
            list of case ids to be loaded from path
        dsetname : str
            key to add the resulting data_dict to the self.datasets atribute
        subjects : str, optional
            category of subjects in dataset, either 'OR' or 'volunteer' by default 'OR'
        
        Raises
        ------
        SystemExit
            if something other than 'OR; or 'volunteer' is passed as subjects
        """
        data_dicts = []
        if subjects == 'volunteer':
            for tdn in names:
                dd = load_volunteer_todict(path, tdn)
                if dd is not None:
                    data_dicts.append(dd)

        elif subjects == 'OR':
            for tdn in names:
                dd = load_OR_todict(path, tdn)
                if dd is not None:
                    data_dicts.append(dd)

        # assemble info from each data dict into a dataset
        caseid = []
        sdata = []
        times = []
        labels = []
        egq = []
        for dd in data_dicts:
            caseid = np.hstack([caseid,np.full(len(dd['t']),dd['name'])])
            dd_sdb = dd['Sdb']

            # check for quality
            if subjects is 'OR':
                dd_egq = dd['egq']
            elif subjects is 'volunteer':
                dd['egq'] = np.ones(len(dd['l']))
                dd_egq = dd['egq']
            else:
                raise SystemExit(f"Error: subjects parameter is incorrect \n subjects is {subjects} but it must be 'volunteer' or 'OR'")
            
            # apply norm
            sdata.append(dd_sdb)
            times += list(dd['t'])
            labels += list(dd['l'])
            egq += list(dd_egq)

        ddf = pd.DataFrame(data = np.hstack([np.array([times]).T, 
                                            np.hstack(sdata).T, 
                                            np.array([egq]).T,
                                            np.array([labels]).T]),
                           columns=np.hstack(['times', dd['f'], 'egq', 'l']))

        ddf['caseid'] = caseid
        ddf = ddf.astype({"egq":'int64',"caseid":'category'})
        # assemble the dict
        dataset_dict = {}
        dataset_dict['fs'] = dd['f']
        dataset_dict['caseid'] = np.array(caseid)
        dataset_dict['labels'] = np.array(labels)
        dataset_dict['times'] = np.array(times)
        dataset_dict['Sdb'] = np.hstack(sdata)
        dataset_dict['subjects'] = subjects
        dataset_dict['egq'] = np.array(egq).astype(bool)
        dataset_dict['df'] = ddf
        self.datasets[dsetname] = dataset_dict

    def featurize(self, t_df, fs, v_df=None, which_features=['Sdb','bands','PCA','LDA','CNN']):
        """Function for generating the features that a classifier actually uses.
        This is called when a classifier is fit and is applied to only the training data.

        returns training_features, pca, lda, validation_features
        TODO
        Parameters
        ----------
        which_features : list (optional)
            which features to featurize
            default is all so ['Sdb','bands','PCA','LDA','CNN']
        
        
        Returns
        -------
        [type]
            [description]
        """

        # filter out signal dropout, unknown labels ONLY FOR FITTING
        good_tdf = t_df[(t_df['l'].isin([0,1])) & (t_df['egq']==1)] 
        
        # do PCA and LDA
        pca_sdb = decomposition.pca.PCA(n_components=3)
        pca_sdb.fit(good_tdf[np.array(fs, dtype='str')].values)
        pca_10 = decomposition.pca.PCA(n_components=10)
        pca_10.fit(good_tdf[np.array(fs, dtype='str')].values)
        lda_sdb = discriminant_analysis.LinearDiscriminantAnalysis(n_components=1)
        lda_sdb.fit(good_tdf[np.array(fs, dtype='str')].values, good_tdf['l'].values)
        if 'CNN' in which_features:
            pca_cnn = fit_pca_cnn(np.unique(good_tdf.caseid))
        else:
            pca_cnn = []

        # temp storage of pca_sdb and lda_sbd
        self.pca_lda = [pca_10, lda_sdb]
        
        # apply to the training data (regardless of label)
        train_feats = self._featurize_df(t_df,which_features,fs,pca_sdb,lda_sdb,pca_cnn)
        
        # check whether there is a validation df to featurize
        if isinstance(v_df, pd.DataFrame): 
            val_feats = self._featurize_df(v_df,which_features,fs,pca_sdb,lda_sdb,pca_cnn)

            return train_feats, val_feats
        else:
            return train_feats

    def _featurize_df(self, df, which_features, fs, pca_sdb, lda_sdb, pca_cnn):
        """
        Helper function to featurize a data frame
        which_features is a list that may contain any of ['Sdb','bands','PCA','LDA','CNN']
        assumes features have already been fit.
        """
        feats_dict = {}
        if 'Sdb' in which_features:
            # Sdb df unchanged
            feats_dict['Sdb'] = df

        if 'bands' in which_features:
            # tb df from bands
            bands = pd.DataFrame(data = Sdb_to_bands(df[np.array(fs, dtype='str')].values, fs))
            dfbands = pd.concat([df['times'], bands, df['egq'], df['l'], df['caseid']], axis=1)
            feats_dict['bands'] =  dfbands

        if 'PCA' in which_features:
            # same for pca
            PCA = pd.DataFrame(data = pca_sdb.transform(df[np.array(fs, dtype='str')].values))
            dfPCA = pd.concat([df['times'], PCA, df['egq'], df['l'], df['caseid']], axis=1)
            feats_dict['PCA'] = dfPCA

        if 'LDA' in which_features:
            # same for LDA
            LDA = pd.DataFrame(data = lda_sdb.transform(df[np.array(fs, dtype='str')].values))
            dfLDA = pd.concat([df['times'], LDA, df['egq'], df['l'], df['caseid']], axis=1)
            feats_dict['LDA'] = dfLDA

        if 'CNN' in which_features:
            # helper fcn for CNN
            dfCNN = apply_pca_cnn(pca_cnn,np.unique(df.caseid))
            feats_dict['CNN'] = dfCNN
        
        return feats_dict

    def add_model(self, ctype='lr', ftype='Sdb', ttype='standard',
                        mname='lr_standard_Sdb'):
        """Adds a model comprised of: a featureset, a classifier, and a method for handling the timeseries.
        Classifier types: 'lr', 'svr'
        Timeseries types: 'standard', 'hmm2', 'hmmfree'
        Feature types: 'Sdb', 'bands', 'PCs', 'LD', 'CNN'

        Parameters
        ----------
        ctype : str, optional
            classifier type, by default 'lr'
        ftype : str, optional
            features for the model, by default 'Sdb'
        ttype : str, optional
            time series treatment, by default 'standard'
        mname : str, optional
            model name, by default 'lr_standard_Sdb'
        """
        # the ultimate classification approach
        if ctype == 'svr':
            model = svm.SVR(kernel='linear', cache_size=8000)
        elif ctype == 'lr':
            model =  linear_model.LogisticRegression()

        # the method for treatment of the timeseries
        model.timeseries = ttype
        if ttype == 'standard':
            model.hmm = None
        elif ttype == 'hmm2':
            hmm = hmm_model.GaussianHMM(2, algorithm='viterbi', n_iter=10)
            model.hmm = hmm
        elif ttype == 'hmmfree':
            hmm = hmm_model.GaussianHMM(6, algorithm='viterbi', n_iter=10)
            model.hmm = hmm
        else:
            raise Exception("The given timeseries treatment does not fall into an approved type (standard, hmm2, hmmfree).")

        # the features fed into the timeseries treatment
        model.ftype = ftype

        # and now append it
        self.models[mname] = model

    def train_model(self, mname, training_dname, timer=False, which_features=['Sdb','bands','PCA','LDA','CNN']):
        """Fits model. The classifier knows which features to take and whether
        or not to HMM it.

        Parameters
        ----------
        mname : str
            model name
        training_dname : str
            dataset to fit on
        timer : bool, optional
            whether we want it trained, by default False
        """

        if timer is True:
            lap = laptimer()

        # collect the data
        model = self.models[mname]
        dataset = self.datasets[training_dname]
        ddf = dataset['df']
        fs = dataset['fs']

        # do the featurization step
        train_feats  = self.featurize(ddf, fs, which_features=which_features)

        # get the specific feature dict
        train_feat = train_feats[model.ftype]

        # drop cols
        nonvalue_cols = ['times', 'egq', 'l', 'caseid']

        # perform the timeseries analysis by taking only eeg quality spots
        if model.timeseries == 'standard':
            # no treatment of the timeseries as a timeseries
            training_series = train_feat[train_feat['egq']==1].drop(nonvalue_cols, axis=1).values
            training_labels = train_feat[train_feat['egq']==1]['l']
            
        else:
            # get the training values from the HMM timeseries
            hmm = model.hmm
            train_lengths = _continuous_lengths(train_feat)
            hmm.fit(train_feat[train_feat['egq']==1].drop(nonvalue_cols, axis=1).values, 
                    train_lengths)
            # calculate posterior probabilities for each state in order to train logistic regression
            posteriors = hmm.score_samples(
                    train_feat[train_feat['egq']==1].drop(nonvalue_cols, axis=1).values, 
                    train_lengths)[1]
            
            # ## calcualte AIC for model parameterized in this way
            # logprob = hmm.decode(train_feat, algorithm='viterbi')[0]
            # n_params = 2*hmm.n_components*hmm.n_features +(hmm.n_components)**2 -1
            # aic = 2*(n_params) - 2*logprob
            # hmm.aic = aic

            training_series = posteriors
            training_labels = train_feat[train_feat['egq']==1]['l']

        # perform training, then get val py
        model.fit(training_series, training_labels)
        model.isfit = True

        # used to featurize the validation data
        model.training_info = [ddf, fs]

        # give the time of the fitting
        if timer is True:
            print(f"Processing time: {np.round(lap(),3)}")

    def crossvalidation(self, mname, dname, n_held=1, timer=False,      
                        features=['Sdb', 'bands', 'PCA', 'LDA', 'CNN']):
        """Performs cross-validation noting that each time, a different n_held must be held out. We want to do every combination available. 

        Parameters
        ----------
        mname : str
            model name
        dname : str
            dataset used for crossval
        n_held : int, optional
            number held out in crossval, by default 1
        timer : bool, optional
            whether to print the time taken to run, by default False
        features : list, optional
            which features should be generated, by default ['Sdb', 'bands', 'PCA', 'LDA', 'CNN']

        Returns
        -------
        list
            crossvalidation AUCs
        """

        # collect the data
        model = self.models[mname]
        dataset = self.datasets[dname]
        ddf = dataset['df']
        fs = dataset['fs']
        
        # get combinations for training / val splits
        unique_trials = np.unique(dataset['caseid'])
        combs = list(combinations(unique_trials, len(unique_trials)-n_held))

        # what we collect from each crossval iteration
        model_performance = [] # AUC over single left out case
        for fold in combs:
            # split for featurization
            train_df = ddf[ddf['caseid'].isin(fold)].reset_index(drop=True)
            val_df = ddf[~ddf['caseid'].isin(fold)].reset_index(drop=True)

            # do the featurization step
            train_feats, val_feats = self.featurize(train_df, fs, v_df=val_df, 
                                                    which_features=features)

            # get the specific feature dict
            train_feat = train_feats[model.ftype]
            val_feat = val_feats[model.ftype]

            # drop cols
            nonvalue_cols = ['times', 'egq', 'l', 'caseid']

            # perform the timeseries analysis by taking only eeg quality spots
            if model.timeseries == 'standard':
                # no treatment of the timeseries as a timeseries
                training_series = train_feat[train_feat['egq']==1].drop(nonvalue_cols, axis=1).values
                training_labels = train_feat[train_feat['egq']==1]['l']
                validation_series = val_feat[val_feat['egq']==1].drop(nonvalue_cols, axis=1).values
                validation_labels = val_feat[val_feat['egq']==1]['l']
                
            else:
                # get the training values from the HMM timeseries
                hmm = model.hmm
                train_lengths = _continuous_lengths(train_feat)
                hmm.fit(train_feat[train_feat['egq']==1].drop(nonvalue_cols, axis=1).values, 
                        train_lengths)
                # calculate posterior probabilities for each state in order to train logistic regression
                posteriors = hmm.score_samples(
                        train_feat[train_feat['egq']==1].drop(nonvalue_cols, axis=1).values, 
                        train_lengths)[1]
                
                # ## calcualte AIC for model parameterized in this way
                # logprob = hmm.decode(train_feat, algorithm='viterbi')[0]
                # n_params = 2*hmm.n_components*hmm.n_features +(hmm.n_components)**2 -1
                # aic = 2*(n_params) - 2*logprob
                # hmm.aic = aic

                training_series = posteriors
                training_labels = train_feat[train_feat['egq']==1]['l']

                val_lengths = _continuous_lengths(val_feat)
                try:
                    val_posteriors = hmm.score_samples_fwd(
                        val_feat[val_feat['egq']==1].drop(nonvalue_cols, axis=1).values,
                        val_lengths)[1]
                    
                except:
                    print('WARNING: You are not using the modified version of HMM learn')
                    print('Your classifier may be using the backward algorithm to predict consciousness')
                    print('This does not affect the performance of the model. It only means this classifier could not be used in real time')
                    print('For access to the forward-only hmmlearn see https://github.com/benyameister/hmmlearn/blob/master/README.rst')
                    val_posteriors = hmm.score_samples(
                        val_feat[val_feat['egq']==1].drop(nonvalue_cols, axis=1).values,
                        val_lengths)[1]
                validation_series = val_posteriors  
                validation_labels = val_feat[val_feat['egq']==1]['l']

            # perform training, then get val py
            model.fit(training_series, training_labels)
            model.isfit = True
            py = model.predict_proba(validation_series)[:,1]

            # save roc from each split
            fpr, tpr = roc_curve(validation_labels, py)[:2]
            auc_split = auc(fpr, tpr)
            
            model_performance.append(auc_split)
        return model_performance
    
    def save_model(self, path, mname):
        """
        Deprecated.
        """
        model = self.models[mname]
        pickle.dump(model, open(path, 'wb'))

    def load_model(self, path, mname):
        """Deprecated."""
        self.models[mname] = pickle.load(open(path, 'rb'))
    

    def validate_model(self, mname, validation_dname, timer=False):
        """applies a model to data within dsetname dataset.
        the resulting predictions are appended to the dataset with mname.
        ensures that norming is same.

        Note that this predicts *everything*, and nan labels must be stripped later (similarly hmm does predict nan labels but not egq=0 because data must be continuous).

        Also note that the features are generated using the PCA or LDA fit from the training data, which is now part of the model.
        """

        if timer is True:
            lap = laptimer()

        # collect the data
        model = self.models[mname]
        dataset = self.datasets[validation_dname]
        ddf = dataset['df']
        fs = dataset['fs']
        ftype=model.ftype

        # ensure model has been fit
        assert model.isfit, "Model is not yet fit!" 
         
        # do the featurization step
        tddf, tfs = model.training_info
        assert all(tfs==fs), "Training/validation frequencies of MTSGs must be same!"

        val_feats = self.featurize(tddf, tfs, v_df=ddf, which_features=[ftype])[1]

        # get the specific feature dict
        val_feat = val_feats[model.ftype]

        # drop cols
        nonvalue_cols = ['times', 'egq', 'l', 'caseid']

        # perform the timeseries analysis by taking only eeg quality spots
        if model.timeseries == 'standard':
            # no treatment of the timeseries as a timeseries
            validation_series = val_feat[val_feat['egq']==1].drop(nonvalue_cols, axis=1).values
            
        else:
            # get the training values from the HMM timeseries
            hmm = model.hmm
            val_lengths = _continuous_lengths(val_feat[val_feat['egq']==1])
            try:
                val_posteriors = hmm.score_samples_fwd(
                    val_feat[val_feat['egq']==1].drop(nonvalue_cols, axis=1).values,
                    val_lengths)[1]
                
            except:
                print('WARNING: You are not using the modified version of HMM learn')
                print('Your classifier may be using the backward algorithm to predict consciousness')
                print('This does not affect the performance of the model. It only means this classifier could not be used in real time')
                print('For access to the forward-only hmmlearn see https://github.com/benyameister/hmmlearn/blob/master/README.rst')
                val_posteriors = hmm.score_samples(
                    val_feat[val_feat['egq']==1].drop(nonvalue_cols, axis=1).values,
                    val_lengths)[1]

            validation_series = val_posteriors

        py = model.predict_proba(validation_series)[:,1]

        # only return wehre we made predictions
        ddf_val = ddf[ddf['egq']==1].copy()
        ddf_val['py'] = py

        # append the validation result to the model itself
        if not hasattr(model, "val_result"):
            model.val_result = {}
        model.val_result[validation_dname] = ddf_val
        model.isval = True
        self.models[mname] = model
        
        # give the time of the fitting
        if timer is True:
            print(f"Processing time: {np.round(lap(),3)}")

    def roc_auc(self, mname, dname, cases='all'):
        """returns fpr, tpr, auc
        strips regions with bad eeg quality or NANs
        """ 
        
        ddf = self.models[mname].val_result[dname]
        if cases=='all':
            validation_l = ddf[ddf['egq']==1]['l'].values
            validation_py = ddf[ddf['egq']==1]['py'].values
            vl = validation_l[~np.isnan(validation_l)]
            py = validation_py[~np.isnan(validation_l)]
        elif type(cases) is list:
            validation_l = ddf[(ddf['egq']==1) & (ddf['caseid'].isin(cases))]['l'].values
            validation_py = ddf[(ddf['egq']==1) & (ddf['caseid'].isin(cases))]['py'].values
            vl = validation_l[~np.isnan(validation_l)]
            py = validation_py[~np.isnan(validation_l)]

        fpr, tpr, thr = roc_curve(vl, py)[:3]
        auct = auc(fpr, tpr)
        sens_plus_spec = (1-fpr)+tpr
        thr_opt = thr[np.argmax(sens_plus_spec)]
        return fpr, tpr, thr, auct, thr_opt

    def acc(self, mname, dname, thr=0.5, cases='all'):
        """returns fpr, tpr, auc
        strips regions with bad eeg quality or NANs
        """ 
        
        ddf = self.models[mname].val_result[dname]
        if cases=='all':
            validation_l = ddf[ddf['egq']==1]['l'].values
            validation_py = ddf[ddf['egq']==1]['py'].values
            vl = validation_l[~np.isnan(validation_l)]
            py = validation_py[~np.isnan(validation_l)]
        elif type(cases) is list:
            validation_l = ddf[(ddf['egq']==1) & (ddf['caseid'].isin(cases))]['l'].values
            validation_py = ddf[(ddf['egq']==1) & (ddf['caseid'].isin(cases))]['py'].values
            vl = validation_l[~np.isnan(validation_l)]
            py = validation_py[~np.isnan(validation_l)]

        ys = py>=thr
        acc = accuracy_score(ys, vl)
        return acc


    def plot_roc(self, mname, dsetname, ax=None, label='', c='b', ls='-'):
        """ plots the roc """
        if ax is None:
            ax = plt.subplot()
        
        fpr, tpr, thr, auct, _ = self.roc_auc(mname, dsetname)
        ax.plot(fpr, tpr, label=f'{label} AUC = {np.round(auct, 3)}', c=c, 
            ls=ls)
        ax.plot([0, 1], [0, 1], 'k:')
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.legend()

    def save_inference_table(self, path, mname, dname):
        """ Deprecated. """
        inf_tab = self.models[mname].val_result[dname]
        inf_tab.to_csv(path+mname+'.csv')



# utility functions

class laptimer:
    """
    Whenever you call it, it times laps.
    """

    def __init__(self):
        self.time = time()

    def __call__(self):
        ret = time() - self.time
        self.time = time()
        return ret

    def __str__(self):
        return "%.3E" % self()

    def __repr__(self):
        return "%.3E" % self()


def save_nparray(filename, nparray, colnames=None):
    """
    Uses pandas to save a numpy array with column headers.
    """
    assert(len(colnames)) == nparray.shape[1], "columns do not match table"
    output_df = pd.DataFrame(data=nparray, columns=colnames)
    output_df.to_csv(filename, index=False)


def save_inference_table(filename, table):
    """helper function for saving inference tables"""
    save_nparray(filename, table, colnames=['case_id', 't', 'p_y', 'y'])


def load_volunteer_todict(path,  name):
    """
    Loads .csv files into a dict. I find this a useful utility.
    """
    f = np.genfromtxt(path + '/' + name + '_f.csv', delimiter=',')
    t = np.genfromtxt(path + '/' + name + '_t.csv', delimiter=',')
    l = np.genfromtxt(path + '/' + name + '_l.csv', delimiter=',')
    Sdb = np.genfromtxt(path + '/' + name + '_Sdb.csv', delimiter=',')
    egq = np.ones(len(t))
    return {'f': f, 't': t, 'Sdb': Sdb, 'l': l, 'name': name, 
            'egq': egq}


def load_OR_todict(path, name):
    """
    Loads .csv files into a dict including events information from CL files
    """
    try:
        f = np.genfromtxt(path + '/' + name + '_f.csv', delimiter=',')
        t = np.genfromtxt(path + '/' + name + '_t.csv', delimiter=',')
        l = np.genfromtxt(path + '/' + name + '_l.csv', delimiter=',')
        Sdb = np.genfromtxt(path + '/' + name + '_Sdb.csv', delimiter=',')
        # events = np.genfromtxt(path +'/' + name + '_events.csv', 
        #                        delimiter=',', usecols=[0, 1], dtype="str")
        egq = np.genfromtxt(path + '/' + name + '_EEGquality.csv', 
                            delimiter=',')
        return_dict = {'f': f, 't': t, 'Sdb': Sdb, 'l':l, #'events': events, 
                'name': name, 'egq':egq}

        # get bolus if it's real
        if os.path.exists(path + '/' + name + '_bolus.csv'):
            bolus = np.genfromtxt(path + '/' + name + '_bolus.csv', 
                                  delimiter=',', dtype='str')
            return_dict['bolus'] = bolus

        # and infusion if it's real
        if os.path.exists(path + '/' + name + '_infusion.csv'):
            infusion = np.genfromtxt(path + '/' + name + '_infusion.csv', 
                                  delimiter=',', dtype='str')
            return_dict['infusion'] = infusion
        
        # and gas if it's real
        if os.path.exists(path + '/' + name + '_gas.csv'):
            gas = np.genfromtxt(path + '/' + name + '_gas.csv', 
                                  delimiter=',', dtype='str')
            return_dict['gas'] = gas
        return return_dict

    except OSError:
        print(f"Incomplete data for {name}")
    except ValueError:
        print(f"Incorrect CSV format for {name}")


def _continuous_lengths(data_df, dt=2):
    """
    calcualtes lengths of continuous observations of eeg data for HMM processing
    This method is written for external validation data which has spontaneous 
    drop out of signal. That is why a 'egq' array is expected from the data _dict
    This method still works for volunteer data, though it is analogous to stacking 
    the lengths of each case

    Parameters
    ----------
    data_df : pandas dataframe 
        should have entries 'egq' and 'caseid'
    dt: float
        spacing in t

    Returns
    -------
    lengths : numpy array
        array of lengths. Each length is an integer number that is the number of continuous samples. 
        Easy check that it is being computed correctly is sum(lengths) = len(goodeeg_times)
    """
    
    lengths = []
    goodeeg_times = data_df[data_df['egq']==1]['times'].values
    length = 1
    for ti, t in enumerate(goodeeg_times[:-1]):
        if goodeeg_times[ti+1]-t==2:
            length+=1
        else:
            lengths.append(length)
            length=1
    lengths.append(length)

    return np.array(lengths)

def hmm_aic(n_states_options,training_data,training_lengths,timer=True,plot=True):
    """
    calcualte AIC for a number of different states based on training data
    to see if there is an ideal number of states for HMMfree
    TODO - params and results
    Parameters
    ----------

    Results
    -------
    """
    AIC = []
    for k in n_states_options:
        if timer is True:
                lap = laptimer()
        model = hmm_model.GaussianHMM(
                            k,
                            algorithm='viterbi',
                            n_iter=10)
        model.fit(training_data.transpose(), training_lengths)
        
        
        logprob = model.decode(training_data.transpose(),algorithm='viterbi')[0]
        n_params = 2*model.n_components*model.n_features +(model.n_components)**2 -1
        aic = aic = 2*(n_params) - 2*logprob
        AIC.append(aic)
        if timer is True and k>15:
                print(f"finished generating model with {k} states")
                print(f"Processing time: {np.round(lap(), 1)} seconds")
    plt.figure(figsize=(4,1.5))
    plt.plot(n_states_options,AIC)
    plt.title("AIC")
    return AIC


def Sdb_to_bands(Sdb, fi):
    """Gets total power in slow-delta, theta, alpha, beta, gamma(?) range.
    """
    S = 10**(Sdb.T/10)

    sr = np.logical_and(fi>=0, fi<=1.)
    dr = np.logical_and(fi>1, fi<4)
    tr = np.logical_and(fi>=4, fi<8)
    ar = np.logical_and(fi>=8, fi<13)
    br = np.logical_and(fi>=13, fi<25)
    gr = np.logical_and(fi>=25, fi<50)

    slow_db  = 10*np.log10(S[sr,:].sum(0)+1E-10)
    delta_db = 10*np.log10(S[dr,:].sum(0)+1E-10)
    theta_db = 10*np.log10(S[tr,:].sum(0)+1E-10)
    alpha_db = 10*np.log10(S[ar,:].sum(0)+1E-10)
    beta_db  = 10*np.log10(S[br,:].sum(0)+1E-10)
    gamma_db = 10*np.log10(S[gr,:].sum(0)+1E-10)
    return np.vstack([slow_db, delta_db, theta_db, alpha_db, beta_db, gamma_db]).T

def fit_pca_cnn(case_ids_train):
    """transform CNN botlenecks into its PC's
    NOTE: only uses volunteer bottlenecks right now because OR ones havent been generated

    Parameters
    ----------
    case_id_train : np.ndarray
        list of integers corresponding to unique case ids 

    
    Returns
    -------
    cnn_pcs : pd.DataFrame
        first 3 principal component of CNN bottleneck values for each window
        also contains times and labels because they don't line up with other features 
    pca_cnn : decomposition.pca.PCA
        fitted PCA to be used for validation data
    """
    btlncks = pd.read_csv(FP_BTLNCKS_VOLUNTEER)

    # filter btlncks to only use cases in training set
    train_case_inds = [case_id in case_ids_train for case_id in btlncks.case_id]
    btlncks = btlncks.loc[train_case_inds,:]

    pca_cnn = decomposition.pca.PCA(n_components=n_pcs_cnn)
    pca_cnn.fit(btlncks.loc[:,'btlnck_0':])
    return pca_cnn

def apply_pca_cnn(pca_cnn, case_ids):
    """calculate principal components for validation data using prefitted pca
    NOTE: only uses volunteer bottlenecks right now because OR ones havent been generated

    Parameters
    ----------
    pca_cnn : decomposition.pca.PCA
        fitted PCA to be used for validation data
    case_id_train : np.ndarray
        list of strings corresponding to unique case ids 

    
    Returns
    -------
    cnn_pcs : pd.DataFrame
        first 3 principal component of CNN bottleneck values for each window
        also contains times and labels because they don't line up with other features
    """
    btlncks = pd.read_csv(FP_BTLNCKS_VOLUNTEER)

    # filter btlncks to only use cases in training set
    train_case_inds = [case_id in case_ids for case_id in btlncks.case_id]
    btlncks = btlncks.loc[train_case_inds,:]
    pcs = pca_cnn.transform(btlncks.loc[:,'btlnck_0':])
    # output as dataframe so it keeps associated t and labels because has different alignment than other features
    column_names = [f'PC{n+1}' for n in range(n_pcs_cnn)]
    cnn_pcs = pd.DataFrame(pcs, columns=column_names)
    cnn_pcs['times'] = btlncks.t.values
    cnn_pcs['l'] = btlncks.is_conscious.values
    cnn_pcs['egq'] =  np.array([1]*len(cnn_pcs['l']))
    cnn_pcs['caseid'] = btlncks.case_id
    return cnn_pcs