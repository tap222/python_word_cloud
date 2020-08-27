import pandas as pd
import preprocessing
import WordCloud
import config
import sentiment
import traceback

Desc = config.desc

def main(pData, Desc, sntmnt=True, wrdcld=True):
    try:
        lstatusPreprocessing,df_process = preprocessing.preprocess(pData,Desc)
        if sntmnt:
            sentiment.sentiment(df_process, Desc)
        if wrdcld:
            WordCloud.plotwordcloud(df_process, Desc)
            
    except Exception as e:
        print('Error ocurred main file')
        print(traceback.format_exc())
        return(-1)
    return (0)

if __name__ == "__main__":
    pData = pd.read_excel('CustRemarks.xlsx')
    main(pData, Desc)
    