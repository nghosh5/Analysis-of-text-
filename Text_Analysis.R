library(syuzhet)
library(sentimentr)
library(xlsx)

library(topicmodels)
library(tidytext)
library(lda)
library(tm)
library(tidyr)
library(ggplot2)
library(dplyr)
library(SnowballC)
library(pander)
library(xlsx)
library(slam)
library(wordcloud)



final <- read.xlsx("C:/Users/indro/Desktop/healthgrade dataset/Healthcare_dataset_comments.xlsx",1)
comments <- data.frame(final)
comments$Comments1 = as.character(comments$Comments1)
emo <- get_nrc_sentiment(comments$Comments1)
nrc<-get_sentiment(comments$Comments1,method = 'nrc')
af<-get_sentiment(comments$Comments1, method="afinn")
binger<-get_sentiment(comments$Comments1, method="bing")
mysenti_nrc_1<- data.frame(comments$Comments1,emo)
mysenti_bing<- data.frame(comments$Comments1,binger)
mysenti_nrc<- data.frame(comments$Comments1,nrc)
mysenti_af<- data.frame(comments$Comments1,af)
mysenti_final <- data.frame(comments$Comments1,binger,nrc,af)

write.csv(mysenti_nrc_1, file="mysenti_nrc_1.csv")
write.csv(mysenti_bing, file="mysenti_bing.csv")
write.csv(mysenti_nrc, file="mysenti_nrc.csv")
write.csv(mysenti_af, file="mysenti_af.csv")
write.csv(mysenti_final, file="mysenti_final.csv")

cname <- file.path("C:", "neg1")   
#cname   
dir(cname) 
docs <- Corpus(DirSource(cname))  
#summary(docs)
#inspect(docs)
#**************************************Preprocessing**************************
docs <- tm_map(docs, tolower) 



#docs <- tm_map(docs, removePunctuation)
#docs <- tm_map(docs, removeNumbers)
#inspect(docs[2])
#docs <- tm_map(docs, removeWords, stopwords("english"))
#docs <- tm_map(docs, stripWhitespace)

stpw <- readLines(file.choose())
custom <- c(stpw,stopwords())


docs <- tm_map(docs, removeWords, c("minute","cancer","family","referred","referral","heard","bodies","honestly","dealt","honest","seriously","examination","examine","close","human","dealt","experienced","experience","closed","bodily","experiencing"))
docs <- tm_map(docs, removeWords, c("","friendly","friend","used","use","uses","problem","doctor","time","son","please","drugs","refill","refills",
                                    "man","office","mother","cant","appointment","reviewed","pay","without",
                                    'will','see','get','just',"mom","dog","ekg","fill","finally","gave",
                                    "regarding","since","still","high","wasted","staff","doesn't",
                                    "type","spend","email","ray","lot","lin","book","physical","rush",
                                    "seemed","people","spent","time","start","think","human","left",
                                    "returned","let","specialist","specialists","sick","completely","got",
                                    "finding","ever","wrong","different","returns","seems","seem",
                                    "questions","felt","new","always","months","time","appointments","friend",
                                    "paid","however","switched","done","friends","infection","else","waste","much",
                                    "problems","extremely","enough","half","things","explain","explained","wouldn't","health","healthcare",
                                    "urgent","question","bill","thorough","answer","answers","able","busy","everything","completed",
                                    "managed","total","symptom","prescribed","prescription","prescriptions","really","believe",
                                    "symptoms","time","return","two","change","way","pressure","charged","find","finding","person",
                                    "manager","managing","times","chaning","happened","provided","possible","message","pocket",
                                    "acceptable","accept","records","absolutely","meds","pay","case","long","basic",
                                    "refer","prescribe","review","provide","based","state","appt","already","yet","treatment","life","babies","immediately","started",
                                    "correctly","switching","switched","entered","write","right","absolute","prescribing","prescribes","other","referring","referrals",
                                    "turned","others","stated","multiple","minutes","mins","treated","appeared","spoke","charged","pays",
                                    "baby","numbers","appears","charges","dismissed","computer","treat","treats","number","disappointed","thing",
                                    "along","alone","performed","","primarily","highly","primary","mistake","performs","drs","drug","drugs","rays","hand","follow","diseases","requests")) 

#docs <- tm_map(docs, PlainTextDocument) 
#*******************************************Document term matrix************************
dtm<- DocumentTermMatrix(docs,control = list( stopwords = custom,stemming = TRUE,
                                              minWordLength = 2, removeNumbers = TRUE, removePunctuation = TRUE))

TDM <- TermDocumentMatrix(docs,control = list( stopwords = custom,stemming = TRUE,
                                               minWordLength = 2, removeNumbers = TRUE, removePunctuation = TRUE))

#*************************USING TF-IDF***************
term_tfidf <- tapply(dtm$v/slam::row_sums(dtm)[dtm$i], dtm$j, mean) *log2(tm::nDocs(dtm)/slam::col_sums(dtm > 0))
summary(term_tfidf)


dtm_new <- dtm[,term_tfidf >= 0.2653]
summary(slam::col_sums(dtm_new))
dtm_new <- dtm_new[row_sums(dtm_new) > 0,]

dtm_new2 = removeSparseTerms(dtm_new, 0.99)
rowTotals <- apply(dtm_new2 , 1, sum) #Find the sum of words in each Document
dtm_new2   <- dtm_new2[rowTotals> 0, ] 
str(dtm_new2)

#*************************************LDA topic model*************************

#Set parameters for Gibbs sampling
burnin <- 4000
iter <- 2000
thin <- 500
seed <-list(2003,5,63,100001,765)
nstart <- 5
best <- TRUE
k=4

ldaOut <-LDA(dtm_new2,k, method="Gibbs", control=list(nstart=nstart, seed = seed, best=best, burnin = burnin, iter = iter, thin=thin))

ldaOut.topics <- as.matrix(topics(ldaOut))
#ldaOut.topics
ldaOut.terms <- as.matrix(terms(ldaOut,5))
ldaOut.terms
#ldaOut.terms <- as.matrix(terms(ldaOut))

#ldaOut.topics <- as.matrix(topics(myModel))
#write.csv(ldaOut.topics,file="ldaOut.topics.csv")
#topicProbabilities <- as.data.frame(ldaOut@gamma)
#write.csv(topicProbabilities,file=paste("LDAGibbs",k,"TopicProbabilities.csv"))

ap_topics <- tidy(ldaOut, matrix = "beta")
ap_topics


dtm <- TermDocumentMatrix(docs)
m <- as.matrix(TDM)
m<- as.matrix(terms(ldaOut,50))
v <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)
head(d, 10)


set.seed(1234)
wordcloud(words = d$word, freq = d$freq, min.freq = 1,
          max.words=200, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"))



#*******************************GGplots****************
ap_top_terms <- ap_topics %>%
  group_by(topic) %>%
  top_n(5, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

ap_top_terms %>%
  mutate(term = reorder(term, beta)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip()




cname <- file.path("C:", "pos1")   
#cname   
dir(cname) 
docs <- Corpus(DirSource(cname))  
#summary(docs)
#inspect(docs)
#**************************************Preprocessing**************************
docs <- tm_map(docs, tolower) 

#toSpace <- content_transformer(function(x, pattern) { return (gsub(pattern, " ", x))})
#docs <- tm_map(docs, toSpace, "-")
#docs <- tm_map(docs, toSpace, "'")
#docs <- tm_map(docs, toSpace, "'")
#docs <- tm_map(docs, toSpace, ".")
#docs <- tm_map(docs, toSpace, """)
#docs <- tm_map(docs, toSpace, """)


stpw <- readLines(file.choose())
custom <- c(stpw,stopwords())

docs <- tm_map(docs, removeWords, c("absolutely","late","end","patel","refer","referred","treated","treats","little","let","couldn't","else","possible","used","extra","leave",
                                    "might","decision","decisions","sense","quality","couldnt","meds","medicines","medicine","medical","yuen","heard","bodies","honestly",
                                    "referring","dealt","seriously","examination","considering","choice","remembers",
                                    "examine","close","human","dealt","experienced","usually","referral",
                                    "experience","closed","bodily","experiencing","minutes","refers","sick","someone","treatment","leaving","sometimes"))


#docs <- tm_map(docs, PlainTextDocument) 
#*******************************************Document term matrix************************
dtm<- DocumentTermMatrix(docs,control = list( stopwords = custom,stemming = TRUE,
                                              minWordLength = 2, removeNumbers = TRUE, removePunctuation = TRUE))


#*************************USING TF-IDF***************
term_tfidf <- tapply(dtm$v/slam::row_sums(dtm)[dtm$i], dtm$j, mean) *log2(tm::nDocs(dtm)/slam::col_sums(dtm > 0))
summary(term_tfidf)


dtm_new <- dtm[,term_tfidf >= 0.3750]
summary(slam::col_sums(dtm_new))
dtm_new <- dtm_new[row_sums(dtm_new) > 0,]

dtm_new2 = removeSparseTerms(dtm_new, 0.99)
rowTotals <- apply(dtm_new2 , 1, sum) #Find the sum of words in each Document
dtm_new2   <- dtm_new2[rowTotals> 0, ] 
str(dtm_new2)


#*************************************LDA topic model*************************

#**********************************LDA 2

#Set parameters for Gibbs sampling
burnin <- 4000
iter <- 2000
thin <- 500
seed <-list(2003,5,63,100001,765)
nstart <- 5
best <- TRUE
k=4

ldaOut_pos <-LDA(dtm_new2,k, method="Gibbs", control=list(nstart=nstart, seed = seed, best=best, burnin = burnin, iter = iter, thin=thin))

ldaOut.topics_pos <- as.matrix(topics(ldaOut_pos))
ldaOut.topics_pos
ldaOut.terms_pos <- as.matrix(terms(ldaOut_pos,5))
ldaOut.terms_pos

#ldaOut.topics_pos <- as.matrix(topics(ldaOut_pos))
#write.csv(ldaOut.topics_pos,file="ldaOut.topics_pos.csv")
#topicProbabilities_pos <- as.data.frame(ldaOut_pos@gamma)
#write.csv(topicProbabilities_pos,file=paste("LDAGibbs",k,"TopicProbabilities_pos.csv"))


ap_topics <- tidy(ldaOut_pos, matrix = "beta")
ap_topics


#*******************************GGplots****************
ap_top_terms <- ap_topics %>%
  group_by(topic) %>%
  top_n(5, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

ap_top_terms %>%
  mutate(term = reorder(term, beta)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip()
