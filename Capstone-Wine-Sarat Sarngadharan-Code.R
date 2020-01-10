#################################################################################
#Author: Sarat Sarngadharan
#Project: Wine Ratings (Choose your own Project as part of HarvardX Capstone)
#This scripts document is show th code that was used to execute the project to 
#reduce the RMSE and come up with a better prediction
#################################################################################

# Load the libraries.

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
library(tidyverse)

if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
library(caret)
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(tm)) install.packages("tm", repos = "http://cran.us.r-project.org")
if(!require(SnowballC)) install.packages("SnowballC", repos = "http://cran.us.r-project.org")
if(!require(wordcloud)) install.packages("wordcloud", repos = "http://cran.us.r-project.org")
if(!require(topicmodels)) install.packages("topicmodels", repos = "http://cran.us.r-project.org")

## Load the required CSV file or analysis from github capstone2 repository

data_wine <- tempfile()
download.file("https://raw.githubusercontent.com/ssarn00/Captsone2/master/winemag-data-130k-v2.csv",data_wine)
ratings_wine <- read.csv(data_wine,   sep = ",", stringsAsFactors = F, row.names = 1,
                    fill = TRUE)
## Initial Exploration of wine dataset
#Summary
summary(ratings_wine)

#evaluate the orgin of the wine (Country / Province)
#Unique Countries in the dataset 
unique(ratings_wine$country)
#Bar Plot of top 5 countries and number of wine ratings
ratings_wine %>% group_by(country) %>% summarize(n=n()) %>% top_n(5,wt=n) %>% 
  ggplot(aes(x=reorder(country,-n),y=n)) +
  geom_bar(stat="identity")+
  geom_text(aes(label=n),color='blue')+
  labs(title="Top 5 Countries", y="No: of Reviews", x="")

#Province
#unique Provinces
length(unique(ratings_wine$province))
##Bar Plot of top 5 provinces and number of wine ratings
ratings_wine %>% group_by(province) %>% summarize(n=n()) %>% top_n(5,wt=n) %>% 
  ggplot(aes(x=reorder(province,-n),y=n)) +
  geom_bar(stat="identity")+
  geom_text(aes(label=n),color='blue')+
  labs(title="Top 5 Provinces", y="No: of Reviews", x="")

#Points Analysis
#Max /Min/ Average Points in the dataset and Standard Deviation  
ratings_wine %>% select(points) %>% 
  summarize(Min_Points=min(points),
            Max_Points=max(points),
            Average_Points=mean(points),
            SD =sd(points))

#Distribution of Points to see if this is normal distribution
ratings_wine %>% select(points) %>% 
  ggplot(aes(points)) +
  geom_histogram(bins = 20)+
  labs(title="Distribution of points", x="Points", y="Count")

#Price Analysis
#Max /Min/ Average Points in the dataset and Standard Deviation  
ratings_wine %>% filter(price != "") %>% 
  summarize(Min_Price=min(price),
            Max_Price=max(price),
            Average_Price=mean(price),
            SD=round(sd(price),2))
#Scatterplot to see th corelation between Price/Points
ratings_wine %>% filter(price !="") %>% 
  ggplot(aes(x=price,y=points))+
  geom_point()+
  labs(title="Price Distribution and Points", y="Points", x="Price")

#corelation
ratings_winep <- ratings_wine %>% filter(price !="") 
cor(ratings_winep$price, ratings_winep$points, method = c("pearson"))

#Taster Analysis
#unique tasters
(unique(ratings_wine$taster_name))
#Number of reviews by top 5 tasters
ratings_wine %>% filter(taster_name != "") %>% 
  group_by(taster_name) %>% summarize(n=n()) %>%top_n(5,wt=n) %>% 
  ggplot(aes(x=reorder(taster_name,-n),y=n)) +
  geom_bar(stat="identity")+
  geom_text(aes(label=n))+
  labs(title="Reviews by Taster", y="Number of reviews", x="")

#Variability in points based on taster pattern (Generous vs Critical tester)
top_tasters <- ratings_wine %>% filter(taster_name!="") %>% group_by(taster_name) %>% summarize(n=n()) %>% top_n(5,wt=n) %>% .$taster_name
ratings_wine %>% filter(taster_name %in% top_tasters) %>% 
  ggplot(aes(x=reorder(taster_name,-points,FUN = median), y=points)) +
  geom_boxplot()+
  labs(title="Top Tasters", y="Points", x="Taster")


#While evaluating the dataset you could see that the title contains the year of make. Would want to analyze if vintage
#series would be rated higher than the newer ones.

#split the string  to remove initial numeric part from winery details
library(stringr)
title_modified<-str_split_fixed(ratings_wine$title," ", 2)

#extract year of make
year_format <- "\\d\\d\\d\\d"
ratings_wine <- ratings_wine %>% mutate(Year_Make = as.numeric(str_extract(title_modified[,2],year_format)))

#update invalid data elemets that has invalid date
ifelse(ratings_wine$Year_Make>2020,"",ratings_wine$Year_Make)
ifelse(ratings_wine$Year_Make<1900,"",ratings_wine$Year_Make)

#Removing bottles for which price is not available

No_Price <- which(is.na(ratings_wine$price))
ratings_wine <- ratings_wine[-No_Price ,]

#Create the training / testing dataset. 
set.seed(1)
test_index <- createDataPartition(y = ratings_wine$points, times = 1, p = 0.1, list = FALSE)
train_set <- ratings_wine[-test_index,]
temp <- ratings_wine[test_index,]

validation <- temp %>% 
  semi_join(train_set, by = "country") %>%
  semi_join(train_set, by = "province") %>%
  semi_join(train_set, by = "taster_name") %>%
  semi_join(train_set, by = "variety") %>%
  semi_join(train_set, by = "winery") %>%
  semi_join(train_set, by = "Year_Make")  


# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
train_set <- rbind(train_set, removed)

rm(removed, temp, test_index)

## RMSE Function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

##################################################################
# Calculate impact of below parameters on the points
# 1) Geographical effect (Country of Origin /Province)
# 2) Variety effect
# 3) Vintage effect (Year of Make)
# 4) Winery effect
# 5) Taster effect
##################################################################

#Naive Implementation with mean points
mean_points_rating<- mean(train_set$points)
#calculate rmse_naive in validation
rmse_naive <- RMSE(validation$points,mean_points_rating)
#store the RMSE data 
rmse_data<- bind_rows(data_frame(Method = "Mean Points", RMSE = rmse_naive))
rmse_data


#Geographical effects - Country 
country_average <- train_set %>% 
  group_by(country) %>% 
  summarize(country_effect = mean(points - mean_points_rating))

predicted_ratings <- mean_points_rating + validation %>%
  left_join(country_average, by='country') %>%
  pull(country_effect)

rmse_country_effect <- RMSE(predicted_ratings, validation$points)
#store the RMSE data 
rmse_data <- bind_rows(rmse_data,
                          data_frame(Method= "Country Effect",
                                     RMSE=rmse_country_effect))
rmse_data

##Geographical effect - Province

province_average <- train_set %>% 
  left_join(country_average, by='country') %>%
  group_by(province) %>%
  summarize(province_effect = mean(points - mean_points_rating- country_effect))

predicted_ratings <- validation %>%
  left_join(country_average, by='country') %>%
  left_join(province_average, by='province') %>%
  mutate(province_effect = mean_points_rating + country_effect + province_effect) %>%
  pull(province_effect)


rmse_province_effect <- RMSE(predicted_ratings, validation$points)
#store the RMSE data 
rmse_data <- bind_rows(rmse_data,
                       data_frame(Method= "Country/Province Effect",
                                  RMSE=rmse_province_effect))
rmse_data


#variety effect
variety_average <- train_set %>% 
  left_join(country_average, by='country') %>%
  left_join(province_average, by='province') %>%
  group_by(variety) %>%
  summarize(variety_effect = mean(points - mean_points_rating - country_effect - province_effect))

predicted_ratings <- validation %>%
  left_join(country_average, by='country') %>%
  left_join(province_average, by='province') %>%
  left_join(variety_average, by='variety') %>%
  mutate(variety_effect = mean_points_rating + country_effect + province_effect + variety_effect) %>%
  pull(variety_effect)

rmse_variety_effect <- RMSE(predicted_ratings, validation$points)
#store the RMSE data 
rmse_data <- bind_rows(rmse_data,
                       data_frame(Method= "Country/Province/Variety Effect",
                                  RMSE=rmse_variety_effect))
rmse_data

#vintage effect
vintage_average <- train_set %>% 
  left_join(country_average, by='country') %>%
  left_join(province_average, by='province') %>%
  left_join(variety_average,by='variety') %>% 
  group_by(Year_Make) %>%
  summarize(vintage_effect = mean(points - mean_points_rating - country_effect - province_effect - variety_effect))


predicted_ratings <- validation %>%
  left_join(country_average, by='country') %>%
  left_join(province_average, by='province') %>%
  left_join(variety_average, by='variety') %>%
  left_join(vintage_average, by='Year_Make') %>%
  mutate(vintage_effect = mean_points_rating + country_effect + province_effect + variety_effect +vintage_effect) %>%
  pull(vintage_effect)

rmse_vintage_effect <- RMSE(predicted_ratings, validation$points)
#store the RMSE data 
rmse_data <- bind_rows(rmse_data,
                       data_frame(Method= "Country/Province/Variety/Vintage Effect",
                                  RMSE=rmse_vintage_effect))
rmse_data

#winery effect
winery_average <- train_set %>% 
  left_join(country_average, by='country') %>%
  left_join(province_average, by='province') %>%
  left_join(variety_average,by='variety') %>% 
  left_join(vintage_average,by='Year_Make') %>% 
  group_by(winery) %>%
  summarize(winery_effect = mean(points - mean_points_rating - country_effect - variety_effect - vintage_effect))

predicted_ratings <- validation %>%
  left_join(country_average, by='country') %>%
  left_join(province_average, by='province') %>%
  left_join(variety_average, by='variety') %>%
  left_join(vintage_average, by='Year_Make') %>%
  left_join(winery_average, by='winery') %>%
  mutate(winery_effect = mean_points_rating + country_effect + province_effect + variety_effect + vintage_effect + winery_effect) %>%
  pull(winery_effect)

rmse_winery_effect <- RMSE(predicted_ratings, validation$points)
#store the RMSE data 
rmse_data <- bind_rows(rmse_data,
                       data_frame(Method= "Country/Province/Variety/Vintage/Winery Effect",
                                  RMSE=rmse_winery_effect))
rmse_data


#taster effect
taster_average <- train_set %>% 
  left_join(country_average, by='country') %>%
  left_join(province_average, by='province') %>%
  left_join(variety_average,by='variety') %>% 
  left_join(vintage_average,by='Year_Make') %>% 
  left_join(winery_average,by='winery') %>% 
  group_by(taster_name) %>%
  summarize(taster_effect = mean(points - mean_points_rating - country_effect - variety_effect - vintage_effect - winery_effect))

predicted_ratings <- validation %>%
  left_join(country_average, by='country') %>%
  left_join(province_average, by='province') %>%
  left_join(variety_average, by='variety') %>%
  left_join(vintage_average, by='Year_Make') %>%
  left_join(winery_average, by='winery') %>%
  left_join(taster_average, by='taster_name') %>%
  mutate(taster_effect = mean_points_rating + country_effect + province_effect + variety_effect + vintage_effect + winery_effect+taster_effect) %>%
  pull(taster_effect)

rmse_taster_effect <- RMSE(predicted_ratings, validation$points)
#store the RMSE data 
rmse_data <- bind_rows(rmse_data,
                       data_frame(Method= "Country/Province/Variety/Vintage/Winery/Taster Effect",
                                  RMSE=rmse_taster_effect))
rmse_data


#NLP on description field
#Reference used: Basic Test mining in R (https://rstudio-pubs-static.s3.amazonaws.com/132792_864e3813b0ec47cb95c7e1e2e2ad83e7.html)

library(tm)
library(SnowballC)
library(wordcloud)

wine_desc_corpus = Corpus(VectorSource(ratings_wine$description))

#preprocessing steps
#  a) change to lower case
#  b) remove punctuation points
#  c) remove stopwords
#  d) remove addition white spaces
#  e) remove numbers

wine_desc_corpus = tm_map(wine_desc_corpus, content_transformer(tolower))
wine_desc_corpus = tm_map(wine_desc_corpus, removePunctuation)
wine_desc_corpus = tm_map(wine_desc_corpus, removeWords, c("the", "and", stopwords("english")))
wine_desc_corpus = tm_map(wine_desc_corpus, stripWhitespace)
wine_desc_corpus = tm_map(wine_desc_corpus, removeNumbers)

#To analyze the textual data, use a Document-Term Matrix (DTM) representation

wine_desc_dtm <- DocumentTermMatrix(wine_desc_corpus)
#To reduce the dimension of the DTM, we can emove the less frequent terms 
wine_desc_dtm = removeSparseTerms(wine_desc_dtm, 0.99)
wine_desc_dtm

#Generate word cloud
freq = data.frame(sort(colSums(as.matrix(wine_desc_dtm)), decreasing=TRUE))
wordcloud(rownames(freq), freq[,1], max.words=50, colors=brewer.pal(9, "Dark2"))


#env
print("Version Info")
version
