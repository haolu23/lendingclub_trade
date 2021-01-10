import pandas as pd
import numpy as np
import mortgage
import sqlalchemy
try:
    from sklearn.externals import joblib
except Exception:
    import joblib
from dateutil.relativedelta import relativedelta
import datetime
import pdb
import re
import xgboost as xgb
import logging
from utils import filters, Dfexpr

class TradingStrategy():
    def __init__(self, lc, html_lc):
        self.lc = lc
        self.html_lc = html_lc

class Node():
    # Base storage structure for the nodes in a Tree object
    def __init__(self, feature, threshold, left_child, right_child, isleaf, prediction):
        self.feature = feature                       # Feature used for splitting the node
        self.threshold = threshold             # Threshold value at the node
        self.left_child = left_child
        self.right_child = right_child
        self.isleaf = isleaf
        self.prediction = prediction

    def split_left(self, x):
        if not self.isleaf:
            left = self.threshold >= x[:, self.feature]
            return left
        else:
            return np.repeat(True, len(x))

class Tree():
    def __init__(self):
        self.root = None

    def _add_node(self, parent, child, is_left):
        if parent is None:
            if self.root is None:
                self.root = child
            else:
                parent = self.root
        else:
            if is_left:
                parent.left_child = child
            else:
                parent.right_child = child

    def _predict(self, node, x):
        if node.isleaf:
            return np.repeat(node.prediction, len(x))
        else:
            prediction = np.zeros(len(x))
            #pdb.set_trace()
            left = node.split_left(x)
            right = np.logical_not(left)
            left_prediction = self._predict(node.left_child, x[left])
            right_prediction = self._predict(node.right_child, x[right])
            prediction[left] = left_prediction
            prediction[right] = right_prediction
            return prediction

    def predict(self, x):
        return self._predict(self.root, x)


class PriceDiscovery(TradingStrategy):
    """
        Strategy: slowly decrease saling price until sold or reach a limit
    """
    def __init__(self, lc, html_lc, maximum_yield, minimum_yield):
        self.super(lc, html_lc)
        self.max_yield = maximum_yield
        self.min_yield = minimum_yield

    def trading_inventory(criteria_file):
        """
            tradable loans are listed notes and my loan inventory meeting
            some criteria
        """
        # get loans listed for sale
        listed_loans = self.html_lc.trading_inventory()
        # get all my notes and find out which to be traded according to critiera rule
        mynotes = pd.DataFrame.from_records(lc.notes()['myNotes'])
        mynotes = filters(mynotes, criteria_file)
        return listed_loans, mynotes[~mynotes.loanId.isin(listed_loans.loanId)]
        
    def traing_cost_adjust(self, loans):
        """
            trading cost is 1% of the listed price
        """
        pass

    def reset_price(self, listed_loans):
        """
            set listed loans' prices
        """
        # get current yield
        # set a new yield
        pass

    def set_new_price(self, loans):
        """
            set initial prices
        parameter
        ---------
        loans: a dataframe of loans returned from lc.notes API

        returns
        -------
        a dataframe with price column
        """
        for i, dat in loans.iterrows():
            m = mortgage.Mortgage()
        return loans


class MarketYield():
    def __init__(self, yield_curve):
        self.yield_curve = yield_curve

    def __call__(self, loans):
        """
        compute recent loan yield curve
        compare loans for sale, their rating, status and ytm
        """
        df = loans[loans.MarkupDiscount < 0]
        df['grade'] = df.LoanClass.str.extract(pat='([A-Z])')
        df = df.merge(self.yield_curve, left_on=['grade', 'RemainingPayments', 'LoanMaturity'], right_on=['grade', 'remaining_payment', 'term'])
        df['ytm_diff'] = df['discount_rate'] - df['YTM']
        # adjust yield based on status
        df['ytm_diff'] = np.where(df['Status'] == 'Current', df['ytm_diff'],
                np.where(df['Status'] == 'In Grace Period', df['ytm_diff'] + 20,
                    np.where(df['Status'] == 'Late (16-30 days)', df['ytm_diff'] + 300,
                        np.where(df['Status'] == 'Late (31-120 days)', df['ytm_diff'] + 700,
                            2000
                            )
                        )
                    )
                )
        return df.sort_values(['DaysSinceLastPayment', 'ytm_diff'])

def fico_adj(loans, scaling_factor=1.0):
    """
    fico score adjustment based on historical loan pricing, which is about 4.5% per 100 fico change
    """
    # scale the adjustment effect
    return -scaling_factor * 0.045 * (loans['FICOEndLow'] - loans['ficoRangeLow'])

def required_ytm( loans):
    """
        compare required default adjusted irr with YTM
    """
    # required yield of 8%
    discount_curve = pd.read_csv("discount_rate_termstructure.csv")
    ytmfunc = MarketYield(discount_curve)
    loans = ytmfunc( loans)
    adj = fico_adj(loans)
    loans['ytm_diff'] = loans['ytm_diff'] + adj.values
    return loans.sort_values('ytm_diff')

def decisiontree_trade( loans):
    """use a decision tree to classify required discounts
        find loans meet or exceed required discounts. then pick based on 
        payment dates and discount
    """
    lastpmnt = np.where(loans.columns == 'DaysSinceLastPayment')[0][0]
    ficolow = np.where(loans.columns == 'FICOEndLow')[0][0]
    neverlate = np.where(loans.columns == 'NeverLate')[0][0]

    left111 = Node(None, None, None, None, True, 0.4)
    left112 = Node(None, None, None, None, True, 0.8)
    left11 = Node(neverlate, False, left111, left112, False, None)
    left12 = Node(None, None, None, None, True, 0.7)
    left1 = Node(ficolow, 690, left11, left12, False, None)
    left21 = Node(None, None, None, None, True, 0.03)
    left22 = Node(None, None, None, None, True, 0.05)
    left2 = Node(ficolow, 690, left21, left22, False, None)
    left = Node(lastpmnt, 20, left1, left2, False, None)
    right = Node(None, None, None, None, True, 0.02)
    root = Node(lastpmnt, 60, left, right, False, None)

    t = Tree()
    t._add_node(None, root, False)
    discounts = t.predict(loans.values)
    loans['discount'] = discounts
    loans['bidprice'] = discounts * loans.OutstandingPrincipal
    loans['spread'] = loans['bidprice'] - loans['AskPrice']

    return loans[loans['spread'] > 0].sort_values(['DaysSinceLastPayment', 'MarkupDiscount'])


def issued_equalent_ytm(loans):
    """
    compute the hazard adjusted irr at issuance date
    use the same irr and remaining cashflow & fico change caused change of expected cashflow to calculate a price
    """
    current_loans = loans[loans.Status=="Current"].copy()
    fair_prices = np.zeros(len(current_loans))
    survprob = pd.read_table("survival_probs.txt")
    seasonality = pd.read_csv("seasonality.csv").dropna()
    survgrp = survprob.groupby(['term', 'grade'])
    seasonadj = mortgage.SeasonalityAdj(seasonality.scale.values, datetime.date.today().month - 1)

    j = 0
    for i, row in current_loans.iterrows():
        fico_change = row['FICOEndLow'] - row['ficoRangeLow']

        # late 16-30
        latele30roi = 0.000741 * fico_change + 0.913007
        # late 30-120
        lategt30roi = 0.000911 * fico_change + 0.971976
        # never late
        neverlateroi = 0.000942 * fico_change + 1.0296

        fico_adj = 0.000942 * (fico_change) + 1 if fico_change < 0 else 1

        if not row['NeverLate']: 
            fico_adj = fico_adj - (neverlateroi - min(latele30roi, lategt30roi)) 

        true_issue_mth = pd.to_datetime(row['issue_d'])
        last_payd = datetime.date.today() - datetime.timedelta(days = row['DaysSinceLastPayment'])
        issue_d = last_payd - relativedelta(months = int(row['LoanMaturity']) - row['RemainingPayments'] - 1)

        # check if the loan has any deferral
        #pdb.set_trace()
        deferral = (issue_d - true_issue_mth.date()).days / 30
        deferral_adj = 1
        if deferral > 1:
        # set deferral loan adjustment
            deferral_adj = 0.98
        #TODO: move out refactor. consider market spread at loan issuing date.
        #adjust for difference between current new loan spread and new loan spread at issuing date.
        current_market_condition_adj = 0.98

        m = mortgage.Mortgage(float(row['intRate']), row['OriginalNoteAmount'], row['OutstandingPrincipal'], 
                          int(row['LoanMaturity']), issue_d, last_payd, row['RemainingPayments'])
        morig = mortgage.Mortgage(float(row['intRate']), row['OriginalNoteAmount'], row['OriginalNoteAmount'], 
                          int(row['LoanMaturity']), issue_d, issue_d)
        service_fee = mortgage.ServiceFee(row['LoanMaturity'])
        prepay_fee = mortgage.PrepaymentFee(row['LoanMaturity'])
        x = survgrp.get_group((row['LoanMaturity'], row['grade']))
        #x = x[x.time > 0]
        survadj = mortgage.SurvivalAdj(x['P(default)'].values, x['P(prepay)'].values, seasonadj)
        survadj_resid_cf =survadj.cashflow(m.cashflow.values * service_fee.fee_discount[row['LoanMaturity'] - row['RemainingPayments']:], 
                          m.principal_cf.cumsum()[::-1].values * prepay_fee.fee_discount[row['LoanMaturity'] - row['RemainingPayments']:])
        survadj_cf = survadj.cashflow(morig.cashflow.values * service_fee.fee_discount, 
                         morig.principal_cf.cumsum()[::-1].values * prepay_fee.fee_discount) 
        survadj_monthly_intrate = np.irr(np.r_[-row['OriginalNoteAmount'], survadj_cf])
        
        fair_prices[j] = current_market_condition_adj * deferral_adj * fico_adj * np.npv(survadj_monthly_intrate, survadj_resid_cf) / (1+survadj_monthly_intrate)**((30-row['DaysSinceLastPayment'])/30)
        j = j+1

    current_loans.loc[:, 'model_price'] = fair_prices

    return current_loans

def grace_period(loans):
    """
    grace period loan price = w * current loan price + (1-w) * (16-30) Late loan price
    """
    df = loans[loans.Status == 'In Grace Period'].copy()
    w = (df.DaysSinceLastPayment - 30) / 16
    w = np.where(w < 0, 0.5, np.where(w > 1, 1, w))
    late_loans = late_roi_predict(df)
    # require extra return to compensate for model risk and uncertainty
    late_loans.loc[:,"model_price"] = 0.8 * late_loans["roi"] * late_loans["OutstandingPrincipal"]

    current_loans = df.copy()
    # fake status in order to call issued_equalent_ytm
    current_loans['Status'] = 'Current'
    current_loans = issued_equalent_ytm(current_loans)
    df['model_price'] = late_loans['model_price'].values * w + (1-w) * current_loans['model_price'].values

    return df

def grace_period_trade(loans):
    UPPER_BOUND = 2.5
    LOWER_BOUND = 1.2
    WEIGHT = 0.4
    df = grace_period(loans)
    df['model_vs_ask_ratio'] = df['model_price'] / df['AskPrice']
    df = df[(df['model_vs_ask_ratio'] > LOWER_BOUND) & (df['model_vs_ask_ratio'] < UPPER_BOUND)]
    # scale the gap between askprice and model_price since the model_price is uncertain
    df['model_adj_discount'] = ((1-WEIGHT) * df['AskPrice'] + WEIGHT*df['model_price']) / (df['OutstandingPrincipal'] + df['AccruedInterest'])
    #pdb.set_trace()
    return df.sort_values('model_adj_discount')

def neverlate(loans, minimum_discount = -1.5, extra_filter=True):
    df = loans[((loans['NeverLate']) & (loans['MarkupDiscount'] < minimum_discount))].copy()
    if extra_filter:
        df = filters(df, "tradeneverlatesole.txt")
    return df.sort_values('MarkupDiscount')

def neverlate_mix_ytm(loans):
    df = required_ytm(neverlate( loans, extra_filter=False))
    df = df[(df['ytm_diff'] < 0)]
    return df.sort_values('MarkupDiscount')

def neverlate_equiv_ytm(loans):
    UPPER_BOUND = 1.5
    LOWER_BOUND = 1
    WEIGHT = 0.4
    df = issued_equalent_ytm(neverlate(loans, extra_filter=False))
    df['model_vs_ask_ratio'] = df['model_price'] / df['AskPrice']
    df = df[(df['model_vs_ask_ratio'] > LOWER_BOUND) & (df['model_vs_ask_ratio'] < UPPER_BOUND)]
    # scale the gap between askprice and model_price since the model_price is uncertain
    df['model_adj_discount'] = ((1-WEIGHT) * df['AskPrice'] + WEIGHT*df['model_price']) / (df['OutstandingPrincipal'] + df['AccruedInterest'])
    #pdb.set_trace()
    return df.sort_values('model_adj_discount')

def current_loan_fico_adj(loans):
    """
    compute fico adjusted roi for current loans
    Linear regression model ROI~delta fico with L1 norm minimization
    linear regression likely under estimated intercept due to large noise. Use L1 norm miminization would be better
    The model only apply to neverlate loans and the regression only used data with decreasing FICO
    """
    loans['fico_change'] = loans.FICOEndLow - loans.ficoRangeLow
    loans['fico_adjust'] = np.where(np.logical_and(loans['NeverLate'], loans['fico_change'] < 0), 0.000942 * loans['fico_change'] + 1.0296,
            np.where(np.logical_and(np.logical_not(loans['NeverLate']), loans.Status=="Current"),
                np.minimum(0.000741 * loans['fico_change'] + 0.913007, # late 16-30
                    # late 30-120
                    0.000911 * loans['fico_change'] + 0.971976), 1))
    return loans

def fico_change_ret( loans):
    df = loans[loans.Status=="Current"]
    df['expected_roi'] = df['fico_adjust']*df.OutstandingPrincipal / df.AskPrice - 1
    return df[df.expected_roi > 0.1].sort_values('expected_roi',ascending=False)

def latebefore_fico_ret( loans):
    """
    calculate expected return base on fico change for loans that are "Current" but was late before
    """
    df = loans[((np.logical_not(loans['NeverLate'])) & (loans.Status=="Current"))]
    # we don't know previous late status, thus conservative
    df['expected_roi'] = df['fico_adjust']*df.OutstandingPrincipal / df.AskPrice - 1
    #pdb.set_trace()
    return df[df.expected_roi > 0.05].sort_values('expected_roi',ascending=False)

def merge_hist(loans):
    """
    merge trading data with historical loan data
    """
    engine = sqlalchemy.create_engine("sqlite:///hist_loans.db")
    # search in historical database
    hist_records = pd.read_sql("select * from loans where id in (%s)" % ",".join([str(x) for x in loans.LoanId]), engine)
    namesmap = pd.read_csv("live2hist_names.csv")
    cols = list(hist_records.columns)
    for i in range(len(cols)):
        if np.any(namesmap['hist'].isin([cols[i]])):
            #pdb.set_trace()
            cols[i] = namesmap['live'][namesmap['hist']==cols[i]].iloc[0]
    hist_records.columns = cols
    loans = loans.merge(hist_records, left_on='LoanId', right_on='id')
    logging.info("%d loans found in historical database" % len(hist_records))
    return loans
 
def live_filter(loans):
    """
    use the same filter as primary market to filter out loans
    and purchase deepest discount loans
    """
    loans = merge_hist(loans)
  
    filtered = filters(loans, "globalfilter.txt")
    # apply same primary loan filters to young loans
    filtered = pd.concat((filtered[filtered.age > 6], filters(filtered[filtered.age <= 6], "primarymktfilter.txt")))
    return filtered

def unique_filter(loans, engine):
    """
    filter out loans already owned
    """
    existing_ids = pd.read_sql("select distinct LoanId id from trade_records where executionStatus='SUCCESS_PENDING_SETTLEMENT' union select distinct LoanId from mock_trades", engine)
    return loans[[x not in existing_ids.id.values for x in loans.LoanId]]


def late_roi_predict(loans):
    """
    predict ROI for late loans
    """
    annual_inc_stat = pd.read_csv("annual_inc_stat.csv")
    df = pd.merge(loans, annual_inc_stat, left_on='zip_code', right_on='zip_code', how='left')
    
    #df['fico_change_bin'] = pd.cut(df.fico_change, bins=np.arange(-340, 160, 40))
    df['credit_hist'] = (pd.to_datetime(df['issue_d']) - pd.to_datetime(df['earliest_cr_line'])) / np.timedelta64(1, 'D') / 365
    df['logcredit_hist'] = np.log(df['credit_hist'])
    df['period_end_lstat'] = df['Status']
    
    rf = joblib.load('recovery_rf/recovery_rf.pkl')
    catcols = joblib.load('recovery_rf/catcols.pkl')
    with open("traderfexp.txt") as f:
        df = Dfexpr(df, f, ['period_end_lstat'])

    # hack, interval datatype cannot be saved to db
    #del df['fico_change_bin']
    for name in catcols:
        if name not in df.columns:
            df[name] = 0

    nullvals = df[catcols].isnull().sum(axis=1)
    nonulls = nullvals < 1

    roi = np.zeros(len(df))
    #pdb.set_trace()
    if np.any(nonulls):
        roi[nonulls] = rf.predict(df[catcols][nonulls])

    # rename column names, remove characters not accepted by db
    df.columns = [re.sub('\(|\)| |\-', '', col) for col in df.columns]
    df['roi'] = roi #rf.predict(df[catcols])
    return df


def late_trade(loans):
    """
    late recovery model, only purchase Late(16-30) and Late(31-120) loans
    """
    princp_discount = loans['AskPrice'] / loans['OutstandingPrincipal']
    # limit the discount to be at least 80%
    df = loans[princp_discount <= 0.2]
 
    df = late_roi_predict(df)

    #pdb.set_trace()

    # require an extra 30% return as a model uncertainty buffer
    df = df[df.roi > 1.3 * df.AskPrice / df.OutstandingPrincipal]

    # only buy one loan a time
    return df.sort_values(by=['roi'], ascending=False).iloc[:1] if len(df) > 0 else df

def issued_trade(loans):
    """
        get freshly issued loans at discount
    """
    princp_discount = loans['AskPrice'] / loans['OutstandingPrincipal']
    df = loans[((loans.Status == "Issued") & (princp_discount <= 0.99))]

    return df.sort_values(by="MarkupDiscount")


def sell_late(lc, loans):
    """
    sell late loans at a discount according to historical recovery rates
    """
    pass
