import config
import WordCloud
import sentiment
import traceback
import visualization
import pandas as pd
import preprocessing

Desc = config.desc
n = config.num
ngram = config.Ngram
pNodeName = config.NodeName

def main(pData, Desc, sntmnt=True, wrdcld=True, viz=True):
    try:
        lstatusPreprocessing,df_process = preprocessing.preprocess(pData, Desc)
        if sntmnt:
            sentiment.sentiment(df_process, Desc)
        if wrdcld:
            WordCloud.plotwordcloud(df_process, Desc)
        if viz:
            visualization.plotmostfrqKwds(df_process, n)
            visualization.plotFreqngram(df_process, n, ngram)
            visualization.plotngramnetwork(df_process, n, pNodeName)
            
    except Exception as e:
        print('Error ocurred main file')
        print(traceback.format_exc())
        return(-1)
    return (0)

if __name__ == "__main__":
    pData = pd.read_excel('CustRemarks.xlsx')
    main(pData, Desc)
    