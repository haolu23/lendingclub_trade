#!/usr/bin/python3
import pandas as pd
import numpy as np
import pdb
import argparse
import sqlalchemy
import time
import os, re
import datetime
import logging
from lendingclub import LendingClub
from utils import filters, Dfexpr, append_db
#from trading_strategy import decisiontree_trade, neverlate, required_ytm
import trading_strategy




def package_loans(loans):
    """create investment string"""
    loanformat = [{'loanId':x['LoanId'], 'noteId':x['NoteId'], 'orderId':x['OrderId'], 'bidPrice':x['AskPrice']} for n, x in loans.iterrows()]
    return loanformat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument('--maxamnt', dest='maxcash',
            action='store', type=float, default=100,
            help='maximum of cash to be invested in this run')
    parser.add_argument('--weight_factor', dest='c',
            action='store', type=float, default=0.0,
            help='a speed factor to determine how new cash weights should be allocated according to model weight and invested cash')
    parser.add_argument('--maxnum', dest='maxnum',
            action='store', type=float, default=2,
            help='maximum number of loans to be invested by each strategy in this run')
    parser.add_argument('--simulate_cash', dest='cash',
            action='store', type=float, default=None,
            help='use given cash amount instead of querying from lendingclub.')
    parser.add_argument('--simulate_with', dest='infile',
            action='store', default=None,
            help='use dumped loan list instead of querying from folio.')
    parser.add_argument('--dry', dest='dry',
            action='store_true', 
            default=False,
            help='enable to only do a dry run.')
    parser.add_argument('--issued', dest='issued',
            action='store_true', 
            default=False,
            help='enable to buy newly issued loans.')

    args = parser.parse_args()
    args.dry = args.dry or args.infile is not None or args.cash is not None

    account_engine = sqlalchemy.create_engine("sqlite:///accounts.db")
    engine = sqlalchemy.create_engine("sqlite:///loans.db")

    logging.basicConfig(format='%(levelname)s:%(asctime)s %(message)s' , level=logging.INFO)

    filtered_loans = None
    accounts = pd.read_sql("select * from accounts a where a.trading_active=1", account_engine)
    #pdb.set_trace()

    for rwid, user in accounts.iterrows():
        lc = LendingClub(int(user['user_id']), user['api_key'], args.dry)
        if args.cash is None:
            account_summary = lc.summary()
            # limit the maximum number of loans to be invested
            available_amnt = min(account_summary['availableCash'], args.maxcash)
            logging.info("buying loans for %s with available amnt %f" % (user['user_name'], available_amnt))
        else:
            available_amnt = args.cash

        if available_amnt >= 0:
            loans = lc.folio_list() if args.infile is None else pd.read_csv(args.infile, index_col=0)
            loans.columns = [re.sub('[ /+]', '', x) for x in loans.columns]
            loans['FICOEndLow'] = pd.to_numeric(loans['FICOEndRange'].str.split('-').str[0])
            loans['YTM'] = pd.to_numeric(loans['YTM'], errors='coerce')
            loans['DateTimeListed'] = pd.to_datetime(loans['DateTimeListed'])
            logging.info("Total loans available: %d" % len(loans))

            # filter out loans that have asking pricer greater than available cash
            filtered_loans = loans[loans.AskPrice <= available_amnt]
            logging.info("After filtering out asking pricer greater than available cash to invest: %d" % len(filtered_loans))

            filtered_loans = filters(filtered_loans, "tradefilter.txt")
            filtered_loans['duration'] = np.maximum(0, filtered_loans['duration'])
            logging.info("After tradefilter: %d" % len(filtered_loans))

            #filtered_loans = filtered_loans[filtered_loans.DateTimeListed > (datetime.date.today() - datetime.timedelta(days=1))]
            # use some same filters in primary market
            # TODO: going to historical database and get more data
            if not args.issued:
                filtered_loans = trading_strategy.live_filter(filtered_loans)
                logging.info("After live filter: %d" % len(filtered_loans))
                filtered_loans = trading_strategy.current_loan_fico_adj(filtered_loans)

            # only take one note from the same loan
            filtered_loans = filtered_loans.sort_values(['LoanId', 'MarkupDiscount']).groupby('LoanId').apply(lambda x: x.iloc[0])
            logging.info("After sorting filter: %d" % len(filtered_loans))
            # filter out loans already traded
            filtered_loans = trading_strategy.unique_filter(filtered_loans, engine)
            logging.info("After unique filter: %d" % len(filtered_loans))
 
            logging.info("%d loans available to be selected after filter" % len(filtered_loans))
            portfolio_map = pd.read_sql("select * from trade_strats where user_id='%s' and weight>0" % user['user_id'], account_engine)
            invested = pd.read_sql("select strategy, sum(bidPrice) invested from trade_records "
                        "where executionStatus='SUCCESS_PENDING_SETTLEMENT' and user='%s' "
                        "group by strategy" % user['user_name'], engine)
            portfolio_map = portfolio_map.merge(invested, left_on='model_name', right_on='strategy', how='left')
            # scale weights
            portfolio_map.fillna(0, inplace=True)
            portfolio_map['weight'] = portfolio_map['weight'] / portfolio_map['weight'].sum()
            portfolio_map['invested'] = portfolio_map['invested'] / portfolio_map['invested'].sum()
            # adjust toward target
            portfolio_map['weight'] = np.minimum(np.maximum(0, args.c * (portfolio_map['weight'] - portfolio_map['invested']) + portfolio_map['weight']), 1.0)
            portfolio_map['weight'] = portfolio_map['weight'] / portfolio_map[portfolio_map['mock']!=1]['weight'].sum()
            portfolio_map['weight'] = np.minimum(portfolio_map['weight'],  1.0)
            portfolio_map.sort_values('weight', inplace=True, ascending=False)
            unused_weight = 0.0
            #pdb.set_trace()
            for idx, row in portfolio_map.iterrows():
                try:
                    logging.info("Trading with strategy %s" % row['model_name'])
                    func = getattr(trading_strategy, row['model_name'])
                    sorted_loans = func(filtered_loans)
                    unused_weight = min(1.0, unused_weight)
                    if len(sorted_loans) > 0:
                        investment_sum = sorted_loans.AskPrice.cumsum()
                        # scale up the weight if previous strategy doesn't invest
                        #pdb.set_trace()
                        investment_amnt = available_amnt * row['weight'] / ((1-unused_weight+1e-6) if unused_weight!=1 else 1)
                        logging.info("Trading with maximum cash amount of %f" % investment_amnt)
                        idx =  np.where(investment_sum > investment_amnt)[0][0] if investment_sum.iloc[-1] > investment_amnt else len(investment_sum)
                        idx = min(idx, args.maxnum)
                        if idx > 0:
                            # make bid
                            if row['mock'] == 1:
                                #sorted_loans.iloc[:idx].to_sql("mock_trades", engine, index=False, if_exists='append')
                                sorted_loans['strategy'] = row['model_name']
                                sorted_loans['time'] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                append_db(sorted_loans.iloc[:idx], "mock_trades", engine)
                            else:
                                bid_result = lc.folio_buy(package_loans(sorted_loans.iloc[:idx]))
                                if 'buyNoteConfirmations' in bid_result:
                                    resultdf = pd.DataFrame(bid_result['buyNoteConfirmations'])
                                    # hacky: executionStatus is a list
                                    resultdf['executionStatus'] = [",".join(x) for x in resultdf['executionStatus']]
                                    resultdf['user'] = user['user_name']
                                    resultdf['strategy'] = row['model_name']
                                    resultdf['time'] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                    append_db(resultdf, "trade_records", engine)
                                    #resultdf.to_sql("trade_records", engine, index=False, if_exists='append')
                                    unused_weight = unused_weight + 1 - resultdf.bidPrice.sum() / investment_amnt
                                logging.info(bid_result)
                                # remove invested loans
                                filtered_loans = filtered_loans[[x not in sorted_loans.iloc[idx:].LoanId for x in filtered_loans.LoanId]]
                        else:
                            logging.info("Not enough cash to be invested under strategy %s" % row['model_name'])
                            unused_weight = unused_weight + (row['weight'] if row['mock'] == 0 else 0)
                    else:
                        logging.info("No loan available to be invested under strategy %s" % row['model_name'])
                        unused_weight = unused_weight + (row['weight'] if row['mock'] == 0 else 0)
                except:
                    pass
            # save a copy of secondary market loans
        if args.infile is None:
            loans.to_csv(os.path.join("folio", "SecondaryMarketAllNotes" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"))
        # wait 5mins for the next batch of loans
        if not args.dry:
            del lc
            t = datetime.datetime.now()
            # folio update loan list every 5mins
            wait_sec = (5 - t.minute % 5) * 60 - t.second
            time.sleep(wait_sec)
