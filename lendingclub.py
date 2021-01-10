import json
import logging
import requests
import pdb
import io
import pandas as pd
#import random

log = logging.getLogger(__name__)

class LendingClub():
    """
        Restful lendingclub API
    """
    VERSION = 'v1'
    LENDINGCLUB_API_URL = 'https://api.lendingclub.com/api/investor/'+VERSION
    def __init__(self, investor_id, api_key, dry_trade=False):
        self.investor_id = investor_id
        self.auth = {'Authorization': api_key, 'Content-Type':"application/json"}
        self.base_url_acct = self.LENDINGCLUB_API_URL+'/accounts/'+str(investor_id)+'/'
        self.base_url_loans = self.LENDINGCLUB_API_URL+'/loans/'
        self.base_url_fund = self.base_url_acct+'funds/'
        self.base_url_folio = self.base_url_acct+'trades/'
        self.base_url_folio_acct = self.base_url_folio + 'accounts/'+str(investor_id)+'/'

        self.old_url_folio = self.LENDINGCLUB_API_URL + '/secondarymarket/'
        
        self.session = requests.Session()
        self.session.headers.update(self.auth)

        self.dry_trade = dry_trade

    def __del__(self):
        self.session.close()
    
    def summary(self):
        response = self.session.get(self.base_url_acct+'summary')
        return response.json()
    
    def loans(self):
        response = self.session.get(self.base_url_loans+'listing')
        return response.json()
    

    def notes(self):
        response = self.session.get(self.base_url_acct+'notes')
        return response.json()
    
    def detailed_notes(self):
        response = self.session.get(self.base_url_acct+'detailednotes')
        return response.json()

    def available_cash(self):
        response = self.session.get(self.base_url_acct+'availablecash')
        return response.json()

    def add_fund(self, amount, transferFrequency='LOAD_NOW', **kwargs):
        required = {'investorId':self.investor_id,
                    'amount':amount,
                    'transferFrequency': transferFrequency}
        required.update(**kwargs)
        response = self.session.post(self.base_url_fund+'add',
                              json.dumps(required))
        return response.json()

    def withdraw_fund(self, amount):
        response = self.session.post(self.base_url_fund+'withdraw',
                              json.dumps({'amount':amount}))
        return response.json()

    def pending_fund(self):
        response = self.session.post(self.base_url_fund+'pending')
        return response.json()

    def cancel_fund(self, transferids):
        response = self.session.post(self.base_url_fund+'cancel',
                json.dumps({"transferIds":transferids}))
        return response.json()

    def portfolio(self):
        response = self.session.get(self.base_url_acct+'portfolios')
        return response.json()

    def invest(self, loans):
        if self.dry_trade:
            return loans
        response = self.session.post(self.base_url_acct+'orders',
                              json.dumps({'aid':self.investor_id, 'orders':loans}))
        return response.json()

    def create_portfolio(self, name, description):
        response = self.session.post(self.base_url_acct+'portfolios',
                json.dumps({'actorId':self.investor_id, 'portfolioName':name, 'portfolioDescription':description}))
        return response.json()

    def folio_list(self):
        response = self.session.get('http://public-resources.lendingclub.com.s3-website-us-west-2.amazonaws.com/SecondaryMarketAllNotes.csv')
        return pd.read_csv(io.BytesIO(response.content))

    def folio_sell(self, notes, expirationDate):
        """
        https://www.lendingclub.com/foliofn/APIDocumentationSell.action
        """
        if self.dry_trade:
            return notes
        response = self.session.post(self.base_url_folio+'sell',
                json.dumps({'Aid':self.investor_id, 'expireDate':expirationDate, 'notes':notes}))
        return response.json()
        
    def folio_buy(self, notes):
        """
        https://www.lendingclub.com/foliofn/APIDocumentationBuy.action
        """
        if self.dry_trade:
            return notes
        response = self.session.post(self.base_url_folio+'buy',
                json.dumps({'aid':self.investor_id, 'notes':notes}))
        return response.json()

    def folio_orders(self, orderType, includeDetails=True):
        response = self.session.post(self.old_url_folio+'accounts/' + str(self.investor_id) + '/orders',
                json.dumps({'includeDetails':includeDetails, 'orderType':orderType}))
        return response#.json()
