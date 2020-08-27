# importing all necessery modules 
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import pandas as pd 
import traceback

def plotwordcloud(pData, Desc):
    try:
        pDescWords = '' 
        # iterate through the csv file 
        for val in pData[Desc]: 
              
            # typecaste each val to string 
            val = str(val) 
          
            # split the value 
            tokens = val.split() 
                    
            pDescWords += " ".join(tokens)+" "
          
        wordcloud = WordCloud(background_color='white',  
                            max_words=300,
                            max_font_size=200, 
                            width=1000, height=800,
                            random_state=42,
                            ).generate(pDescWords) 
          
        # plot the WordCloud image                        
        print(wordcloud)
        fig = plt.figure(figsize = (12,14))
        plt.imshow(wordcloud)
        plt.title(" Custtomer Remarks",fontsize=25)
        plt.axis('off')
        #plt.show()
        plt.savefig('wordcloud.png')
    except Exception as e:
        print('Error ocurred wordcloud file')
        print(traceback.format_exc())
        return(-1)
    return (0)