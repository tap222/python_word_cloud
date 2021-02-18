import traceback
import pandas as pd 
import matplotlib.pyplot as plt 
from wordcloud import WordCloud, STOPWORDS 

def plotwordcloud(pData, Desc, filename):
    try:
        pDescWords = '' 
        # iterate through the csv file 
        for val in pData.Sample: 
              
            # typecaste each val to string 
            val = str(val)
          
            # split the value 
            tokens = val.split()
                    
            pDescWords += " ".join(tokens)+" "
          
        wordcloud = WordCloud(background_color='white',  
                            max_words=300,
                            max_font_size=200, 
                            width=1000, height=800,
                            random_state=42,collocations=False
                            ).generate(pDescWords) 
          
        # plot the WordCloud image                        
        fig = plt.figure(figsize = (12,14))
        plt.imshow(wordcloud)
        plt.title(str(filename), fontsize=25)
        plt.axis('off')
        #plt.show()
        plt.savefig('./output/wordcloud' + str(filename) + '.png')
        
    except Exception as e:
        print('Error ocurred wordcloud file')
        print(traceback.format_exc())
        return(-1)
    return (0)