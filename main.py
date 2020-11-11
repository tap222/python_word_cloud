import config #any hardcoded value defined from config
import WordCloud #Wordcloud scipt for word cloud presentation
import sentiment #Sentiment value each ticket description
import visualization #it is represnting top keywords 
import preprocessing # pre-processing file

import traceback 
import pandas as pd

Desc = config.desc
n = config.num
ngram = config.Ngram
pNodeName = config.NodeName
pSubUnit = config.subunit
pFileName = config.FileName

def main(pData, Desc, sntmnt=False, wrdcld=True, viz=True, Market=False):
    try: 
        lstatusPreprocessing,pDataProcess = preprocessing.preprocess(pData, Desc)
        if Market:
            pMarketUnitList = pDataProcess[pSubUnit].unique().tolist()
            for index in range(len(pMarketUnitList)):
                pMarketUnitData = pDataProcess.loc[pDataProcess[pSubUnit] == pMarketUnitList[index]]
                if sntmnt:
                    sentiment.sentiment(pMarketUnitData, Desc = pDataProcess['Sample'], filename=str(pMarketUnitList[index]))
                if wrdcld:
                    WordCloud.plotwordcloud(pMarketUnitData, Desc = pDataProcess['Sample'], filename=str(pMarketUnitList[index]))
                if viz:
                    visualization.plotmostfrqKwds(pMarketUnitData, n, filename=str(pMarketUnitList[index]))
                    visualization.plotFreqngram(pMarketUnitData, n, ngram, filename=str(pMarketUnitList[index]))
                    visualization.plotngramnetwork(pMarketUnitData, pNodeName, filename=str(pMarketUnitList[index]))
        else:
            if sntmnt:
                sentiment.sentiment(pDataProcess, Desc = pDataProcess['Sample'], filename = 'AllData')
            if wrdcld:
                WordCloud.plotwordcloud(pDataProcess, Desc = pDataProcess['Sample'], filename = 'AllData')
            if viz:
                visualization.plotmostfrqKwds(pDataProcess, n, filename = 'AllData')
                visualization.plotFreqngram(pDataProcess, n, ngram, filename = 'AllData')
                visualization.plotngramnetwork(pDataProcess, pNodeName, filename = 'AllData')
                visualization.plottablefrqKwds(pDataProcess, n, filename = 'AllData')              
            
    except Exception as e:
        print('Error ocurred main file')
        print(traceback.format_exc())
        return(-1)
    return (0)

if __name__ == "__main__":
    pData = pd.read_excel(pFileName)
    main(pData, Desc)
    