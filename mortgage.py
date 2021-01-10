import datetime
import pandas as pd
import scipy.optimize as opt
import numpy as np
from enum import Enum
import pdb

# TODO:
# 1. how to handle curtailment
#

class Mortgage(object):
    """
    A mortgage object for calculating mortgage payments,
    yield rates etc.
    Only consider loans that are current, not able to analysis delinquent loans
    @param note a pandas series note object constructed from detailednote API query
    """
    def __init__(self, intrate, initial_principal, remaining_principal, 
            term,
            issue_date, last_payment_date, remaining_payments=None):
        ## lendingclub noteId
        ## origination date
        #self.note = note
        ## monthly interest rate in percentage
        #self.rate = note['interestRate'] / 12 / 100
        self.rate = intrate / 12 / 100
        self.remaining_principal = remaining_principal
        self.remaining_payments = remaining_payments
        self.term = term
        ## last payment date
        #self.last_payment_date = pd.to_datetime(note['lastPaymentDate'] if note['lastPaymentDate'] else note['issueDate'])  
        self.last_payment_date = last_payment_date
        ## monthly_payment
        #self.monthly_payment = note['noteAmount'] * self.__monthly_payment(self.rate, note['loanLength'])
        self.monthly_payment = initial_principal * self.__monthly_payment(self.rate, term)

        ##orig_scheduled_paydates = pd.date_range(pd.to_datetime(note['issueDate']), periods=note['loanLength'], 
        ##        freq='M').shift(1, freq=pd.datetools.monthEnd)
        #scheduled_paydates = pd.date_range(self.last_payment_date, periods=note['loanLength'], 
        #        freq='M').shift(self.last_payment_date.day, freq=pd.datetools.day)
        #scheduled_paydates = pd.date_range(self.last_payment_date, periods=term, 
        #        freq='M').shift(self.last_payment_date.day, freq=pd.datetools.day)
        scheduled_paydates = pd.date_range(issue_date, periods=term, 
                freq='M')
        ## remaining scheduled payment dates
        #self.scheduled_paydates = scheduled_paydates#[scheduled_paydates <= orig_scheduled_paydates[-1]]
        self.scheduled_paydates = scheduled_paydates#[scheduled_paydates <= orig_scheduled_paydates[-1]]

    @staticmethod
    def __monthly_payment(rate, term):
        return rate * (1 + rate)**term / ((1 + rate)**term - 1)

    def accured(self):
        # accured interest
        return self.rate * \
            (datetime.datetime.today() - self.last_payment_date) / 365

    @property
    def principal_cf(self):
        if '_principal_cf' not in self.__dict__:
            x = self.cashflow
        return self._principal_cf


    @property
    def interest_cf(self):
        if '_interest_cf' not in self.__dict__:
            x = self.cashflow
        return self._interest_cf

    @property
    def cashflow(self):
        """
        project cashflow
        """
        if '_cashflow' in self.__dict__:
            return self._cashflow
        
        #scheduled_payments = pd.Series(self.monthly_payment, index=self.scheduled_paydates)
        # calculate principal and interest portions of the payments
        starting_principal = self.remaining_principal#note['principalPending']
        interest_portion = []
        principal_portion = []
        idx = 0
        if self.remaining_payments is None:
            while True:
                interest = starting_principal * self.rate
                interest_portion.append(interest)
                idx = idx + 1
                if self.monthly_payment - interest > starting_principal or \
                    np.isclose(self.monthly_payment, interest + starting_principal, 0.01):
                    principal = starting_principal
                    principal_portion.append(principal)
                    break
                else:
                    principal = self.monthly_payment - interest
                    principal_portion.append(principal)
                starting_principal = starting_principal - principal
        else:
             for i in range(self.remaining_payments - 1):
                interest = starting_principal * self.rate
                interest_portion.append(interest)
                principal = self.monthly_payment - interest
                principal_portion.append(principal)
                starting_principal = starting_principal - principal

             interest = starting_principal * self.rate
             interest_portion.append(interest)
             principal_portion.append(starting_principal)
             idx = self.remaining_payments

        # hacky, idx must be a data error
        if idx < 0:
            interest_portion = pd.Series([])
            principal_portion = pd.Series([])
        else:
            interest_portion = pd.Series(interest_portion, index=self.scheduled_paydates[self.term-idx:])
            principal_portion = pd.Series(principal_portion, index=self.scheduled_paydates[self.term-idx:])

        self._interest_cf = interest_portion
        self._principal_cf = principal_portion
        # curtailment adjustment
        self._cashflow = interest_portion + principal_portion
        return self._cashflow

    def yield_rate(self, ask_price):
        """
        calculate yield based on ask price
        """
        cf = self.cashflow
        # accured, TODO: normalize to month
        next_payment_distance = (cf.index[0].date() - datetime.date.today()).days/30.0
        op = opt.root(lambda r: np.npv(r, cf)/(1 + r)**next_payment_distance - ask_price, self.rate)
        #pdb.set_trace()
        return op.x * 12

    def price(self, annual_yield):
        """
        given discount rate, calculate NPV
        """
        cf = self.cashflow
        # accured, TODO: normalize to month
        next_payment_distance = (cf.index[0].date() - datetime.date.today()).days/30.0
        return np.npv(annual_yield, cf)/(1+annual_yield)**next_payment_distance

class ServiceFee():
    """
    lendingclub charge a constant service fee of 1% of payments
    """
    def __init__(self, term):
        self.fee_discount = np.repeat(0.99, term)


class PrepaymentFee():
    """
    lendingclub charge prepayment fee of 1% after 1yr
    """
    def __init__(self, term):
        NO_PREPAY_PENALTY_PERIOD = 12
        self.fee_discount = np.r_[np.ones(NO_PREPAY_PENALTY_PERIOD), np.repeat(0.99, term-NO_PREPAY_PENALTY_PERIOD)]


class SurvivalAdj():
    def __init__(self, default_cump, prepay_cump, default_seasonality=None):
        if len(default_cump) != len(prepay_cump):
            raise RuntimeError("default hazard and prepay hazard length must equal")
        self.default_hazard = np.diff(default_cump)
        self.prepay_hazard = np.diff(prepay_cump)
        self.term = len(self.default_hazard)
        self.default_seasonality = default_seasonality

    def cashflow(self, contract_cf, prepayment_principals, recovery_ratio=0.0):
        """
        return the hazard rates adjusted cashflow
        contract_cf: fee adjusted contract cashflow
        prepayment_principals: prepayment fee adjusted principal
        """
        if len(contract_cf) > self.term:
            raise RuntimeError("cashflow longer than loan term")
        remaining_cf = len(contract_cf)
        adj_default_hazard = self.default_hazard[self.term-remaining_cf:]
        if (self.default_seasonality is not None):
            adj_default_hazard = self.default_seasonality.hazard(adj_default_hazard)
        adj_prepay_hazard = self.prepay_hazard[self.term-remaining_cf:]
        survival_p = (1-adj_default_hazard-adj_prepay_hazard).cumprod()
        return contract_cf * survival_p + adj_prepay_hazard * prepayment_principals + recovery_ratio * adj_default_hazard

class SeasonalityAdj():
    """
        Adjust default hazard rate by seasonality
    """
    def __init__(self, default_seasonality, starting_month):
        self.seasonality = default_seasonality
        self.starting_month = starting_month

    def hazard(self, default_hazards):
        #cycles = len(default_hazards) // len(self.seasonality)
        #residual_mths = len(default_hazards) - cycles * len(self.seasonality)
        end_mth = self.starting_month+len(default_hazards)
        repeats = end_mth // len(self.seasonality) + 1

        adjs = np.tile(self.seasonality, repeats)[self.starting_month:end_mth]
        return adjs * default_hazards


class LoanStatus(Enum):
    Current = 1
    Grace = 2 
    LateOneMonth = 3
    LateTwoMonths = 4
    Default = 5
    ChargedOff = 6
