from textblob import TextBlob
import traceback
import pandas as pd

def sentiment(pData, Desc):
    try:
        reindexed_data = pData['Sample']
        reindexed_data.index = pData[Desc]

        blobs = [TextBlob(reindexed_data[i]) for i in range(reindexed_data.shape[0])]
        polarity = [blob.polarity for blob in blobs]
        subjectivity = [blob.subjectivity for blob in blobs]

        sentiment_analysed = pd.DataFrame({'Sample':reindexed_data, 
                                           'polarity':polarity, 
                                           'subjectivity':subjectivity,
                                           'Remarks':reindexed_data.index},
                                          index=reindexed_data.index)
        sentiment_analysed.to_excel('AnalyzedData.xlsx',index=False)
    except Exception as e:
        print('Error ocurred due to template')
        print(traceback.format_exc())
        return(-1)
    return (0)