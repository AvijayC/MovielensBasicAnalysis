### MOVIELENS CODE ###
# Author: Avijay Chakravorti
# Purpose: Run RMSE analysis and generate predictions on movielens validation set.

############################################
### SECTION 1: DATA DOWNLOAD AND LOADING ###
############################################
# Loading libraries:
library(tidyverse)
library(caret)
library(data.table)
library(dplyr)
library(tidyr)
library(lubridate)
library(recommenderlab)

# We will be using the provided course code, inside an if statement.
# The if statement will check if the downloaded files exist. If they do not, it will redownload them using the course code.

if( file.exists("edx.rds")==FALSE | file.exists("validation.rds")==FALSE ) {
  #---------------------------------------------------------
  # Create edx set, validation set (final hold-out test set)
  #---------------------------------------------------------
  
  if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
  if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
  if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
  
  library(tidyverse)
  library(caret)
  library(data.table)
  
  # MovieLens 10M dataset:
  # https://grouplens.org/datasets/movielens/10m/
  # http://files.grouplens.org/datasets/movielens/ml-10m.zip
  
  dl <- tempfile()
  download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
  
  ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                   col.names = c("userId", "movieId", "rating", "timestamp"))
  
  movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
  colnames(movies) <- c("movieId", "title", "genres")
  
  # if using R 4.0 or later:
  movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                             title = as.character(title),
                                             genres = as.character(genres))
  
  movielens <- left_join(ratings, movies, by = "movieId")
  
  # Validation set will be 10% of MovieLens data
  set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
  test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
  edx <- movielens[-test_index,]
  temp <- movielens[test_index,]
  
  # Make sure userId and movieId in validation set are also in edx set
  validation <- temp %>% 
    semi_join(edx, by = "movieId") %>%
    semi_join(edx, by = "userId")
  
  # Add rows removed from validation set back into edx set
  removed <- anti_join(temp, validation)
  edx <- rbind(edx, removed)
  
  rm(dl, ratings, movies, test_index, temp, movielens, removed)
  
  ## Storing edx as original format, and dataframe. 
  
  edx_df <- as.data.frame(edx)
  validation_df <- as.data.frame(validation)
  saveRDS(edx, file="edx.rds")
  saveRDS(validation, file="validation.rds")
  
} else {
  # If the files are present, read and load them. 
  # Convert them to dataframes.
  
  edx <- readRDS("edx.rds")
  validation <- readRDS("validation.rds")
  edx_df <- as.data.frame(edx)                # Dataframe version of edx.
  validation_df <- as.data.frame(validation)  # Dataframe version of validation.
  
}


###################################
### SECTION 2: VIEWING THE DATA ###
###################################
# We want to look at the heads and structure of the data.

rm(edx)                   # Unload the edx dataset, we have edx_df and want to save RAM.
rm(validation)            # Same, but for validation dataset.
gc()                      # You will see this many times, it bascially frees up more memory.

# Seeing what edx_df looks like.
head(edx_df) 
length(edx_df)
nrow(edx_df)

# Seeing what validation_df looks like.
head(validation_df)
length(validation_df)
nrow(validation_df)


#############################################################################
### SECTION 3: MODIFYING / MUTATING DATA TO REPRESENT GENRES INDIVIDUALLY ###
#############################################################################

# Currently, data only has 1 column with all relevant genres for that movie in a single string.
# We want to convert that genre string into a vector of 0's and 1's for every row. 
# A 0 means that genre is not in the movie, a 1 means that genre is in the movie.
# There are 20 unique genres, so the vector will be 20 n big.
# We will place this vector into the edx_df and validation_df dataframes, as 20 additional columns added to each row.

# Get all genres from unique applied to separated edx$genres.
# We can use this for the validation set as well. We assume the edx dataset has all unique genres in the validation set.
unique_genres <- unique(scan(text=edx_df$genres, what="", sep="|"))

# Define empty vectors to hold column names
genre_binary_header_idx <- c()
usergenre_binary_header_idx <- c()


# Make the column names.
for (k in 1:length(unique_genres)) {
  temp_col <- sprintf("g_%i", k)
  usertemp_col <- sprintf("ug_%i", k)
  genre_binary_header_idx <- append(genre_binary_header_idx, temp_col)
  usergenre_binary_header_idx <- append(usergenre_binary_header_idx, usertemp_col)
}

genre_binary_header_idx
usergenre_binary_header_idx

# Now that we have our column names, we can begin generating our 20-long binary vectors and assigning them to the rows of edx_df.

##===================================
## SUBSECTION 3.1: MUTATING EDX_DF ##
##===================================

# Split genres column into a list of genres into column named genres_list
edx_df <- edx_df %>% mutate(genres_list = as.list(str_split(genres, "\\|")) )

# Mutate UNIX timestamp into date, month, etc. using lubridate package.
# The weekend column says if the review was generated on a weekend (1) or weekday (0)
# The quarter column gives the fiscal quarter the review took place in.
# Year column is self-explanatory.
#
edx_df <- edx_df %>% mutate(date = as_datetime(timestamp)) %>%
  mutate(quarter = as_factor(quarter(date)),
  month_num = month(date),
  year = year(date),
  weekend =
    ifelse(weekdays(date)%in%c("Friday", "Saturday", "Sunday"),1,0) )

head(edx_df)


## Generating column binary matrix.
edx_df_genres_list <- lapply(edx_df[, "genres_list"], function(given_row) {
  unique_genres %in% unlist(given_row)
})

# Convert the list of lists into a matrix of lists (1 column)
edx_df_genres_list <- as.matrix(edx_df_genres_list)

# Convert the vector of lists into a matrix that has same number of rows as edx_df, 1 col. for every genre.
edx_df_genres_mat <- matrix(unlist(edx_df_genres_list), nrow=nrow(edx_df), length(unique_genres), byrow=TRUE)
# Add column names to the matrix.
colnames(edx_df_genres_mat) <- genre_binary_header_idx
edx_df_genres_mat[1:2, ]

# Join the genre matrix with the dataframe edx_df.
edx_df[, genre_binary_header_idx] <- edx_df_genres_mat*1 # Multiply by 1 to make into vector of 0 and 1.
head(edx_df, 2)

##==========================================
## SUBSECTION 3.2: MUTATING VALIDATION_DF ##
##==========================================

# Split genres column into a list of genres into column named genres_list
validation_df <- validation_df %>% mutate(genres_list = as.list(str_split(genres, "\\|")) )

# Mutate UNIX timestamp into date, month, etc. using lubridate package.
# The weekend column says if the review was generated on a weekend (1) or weekday (0)
# The quarter column gives the fiscal quarter the review took place in.
# Year column is self-explanatory.
#
validation_df <- validation_df %>% mutate(date = as_datetime(timestamp)) %>%
  mutate(quarter = as_factor(quarter(date)),
         month_num = month(date),
         year = year(date),
         weekend =
           ifelse(weekdays(date)%in%c("Friday", "Saturday", "Sunday"),1,0) )

head(validation_df)


## Generating column binary matrix.
validation_df_genres_list <- lapply(validation_df[, "genres_list"], function(given_row) {
  unique_genres %in% unlist(given_row)
})

# Convert the list of lists into a matrix of lists (1 column)
validation_df_genres_list <- as.matrix(validation_df_genres_list)

# Convert the vector of lists into a matrix that has same number of rows as validation_df, 1 col. for every genre.
validation_df_genres_mat <- matrix(unlist(validation_df_genres_list), nrow=nrow(validation_df), length(unique_genres), byrow=TRUE)
# Add column names to the matrix.
colnames(validation_df_genres_mat) <- genre_binary_header_idx
validation_df_genres_mat[1:2, ]

# Join the genre matrix with the dataframe validation_df.
validation_df[, genre_binary_header_idx] <- validation_df_genres_mat*1 # Multiply by 1 to make into vector of 0 and 1.
head(validation_df, 2)

# Now we have both dataframes with their multi-hot genre vectors.
# Delete the placeholder matrices.
rm(edx_df_genres_list)
rm(edx_df_genres_mat)
rm(validation_df_genres_list)
rm(validation_df_genres_mat)
gc()


#############################################################
### SECTION 4: INITIAL DATA VISUALIZATION AND CORRELATION ###
#############################################################

# PLOT: Histogram of number of ratings per movie.
plot_hist_numRatingsPerMovie <- edx_df %>% dplyr::count(movieId) %>%
  ggplot(aes(n)) + geom_histogram(bins=40) + scale_x_log10() + 
  ggtitle("Histogram - Number of Ratings a Movie Recieves")
plot_hist_numRatingsPerMovie

# PLOT: Histogram of number of ratings per user.
plot_hist_numRatingsPerUser <- edx_df %>% dplyr::count(userId) %>%
  ggplot(aes(n)) + geom_histogram(bins=40) + scale_x_log10() + 
  ggtitle("Histogram - Number of Ratings a User Makes")
plot_hist_numRatingsPerUser

# TABLE: Ratings of weekends vs non-weekends
edx_df %>% group_by(rating) %>% 
  summarize(prop_weekend = mean(weekend), n = n())
cat('Correlation between weekend and rating: ', 
    cor(edx_df$weekend, edx_df$rating))
# We see very loose positive correlation between prop_weekend and rating. 
# Not significant. Will not use this. 

# TABLE: Ratings vs. month.
edx_df %>% group_by(month_num) %>%
  summarize(avg_rating = mean(rating), n=n())
cat('Correlation between rating and month_num: ', 
    cor(edx_df$month_num, edx_df$rating))
# Basically no correlation between month_num and avg_rating.
# Not significant. Will not use this.
gc()

# It turns out the dates are not really useful for predicting ratings. Just ignore those columns, we won't use them later.



####################################################
### SECTION 5: TRAIN TEST TUNE SPLIT FROM EDX_DF ###
####################################################

# Split edx_df into train_all and test set.
test_index <- createDataPartition(y = edx_df$rating, times = 1,
                                  p = 0.10, list = FALSE)
train_all_set <- edx_df[-test_index,]
test_set <- edx_df[test_index,]

# Seeing dims of train_all set
nrow(train_all_set)
# Seeing dims of test set
nrow(test_set)

# We need to FURTHER split the train_all set into a train and tune set.

# tune_set definition: We may need a subset of the training set for use in lambda parameter tuning.

tune_index <- createDataPartition(y = train_all_set$rating, times=1, 
                                  p = 0.05, list=FALSE)
train_set <- train_all_set[-tune_index,]
tune_set <- train_all_set[tune_index,]

# Seeing dims of train set
nrow(train_set)
# Seeingdims of tune set
nrow(tune_set)

# Deleting the now defunct train_all set. We have train, tune and test sets already. Don't need this anymore.
rm(train_all_set)
gc()


###############################################
### SECTION 6: COURSE 8 STATISTICAL METHODS ###
###############################################

# We will be repeating what we did for course 8, to see how those methods perform on this dataset.
# This also gives a good benchmark to beat with our own method later.

# For the train_set, get the average rating per movieId and average rating per userId
# Define a function to do this for a given dataset of the same shape as edx_df.

append_user_movie_avgs_preds <- function(train_df ,some_df) {
  # Purpose: Center the ratings around the mean rating of the df. Then,
  # add the user's mean rating, movie's mean rating, and both to the centered rating (rating_sym).
  # Then, calculate error comparing the predicted centered ratings with the centered rating rating_sym.
  # Append all of this to the df and return it.
  
  df_mean <- mean(train_df$rating) # Average rating across train dataframe
  ratings_per_movieId <- train_df %>% group_by(movieId) %>% summarize(avg_movie_bias = mean(rating)-df_mean) # How much a movie's avg rating deviates from global average.
  ratings_per_userId <- train_df %>% group_by(userId) %>% summarize(avg_user_bias = mean(rating)-df_mean) # How much a user's avg rating deviates from global average.
  
  # Merge the movie and user deviations to the main dataset.
  final_df <- base::merge(some_df, ratings_per_movieId, by="movieId", all.x=TRUE) 
  final_df <- base::merge(final_df, ratings_per_userId, by="userId", all.x=TRUE)
  
  # Make predictions based on user average rating, movie average rating, and both combined.
  final_df <- final_df %>% mutate(rating_sym = rating-df_mean) %>% # Rating minus the average rating = rating_sym
    mutate(pred_useronly=avg_user_bias+df_mean, 
           pred_movieonly=avg_movie_bias+df_mean, 
           pred_both=avg_movie_bias+avg_user_bias+df_mean) %>%
    mutate(mean_rating = df_mean)
  
  # Remember, we can only use the ratings in the train set. We cannot use ratings from any other set for predictions.
  
  # Print column names to see what the new dataset is like.
  cat(colnames(final_df), "\n")
  final_df
}


# We call the above function on all dataframes to get additional columns:
# [avg_movie_bias avg_user_bias rating_sym pred_useronly pred_movieonly pred_both err_useronly err_movieonly err_both]
# The "ratings" data MUST ONLY COME FROM THE TRAIN SET. So, the first argument is always the train set.
gc()

train_set <- append_user_movie_avgs_preds(train_set, train_set) # Predict train set from train_set data
gc() 
test_set <- append_user_movie_avgs_preds(train_set, test_set) # Predict test set from train_set data
tune_set <- append_user_movie_avgs_preds(train_set, tune_set) # Predict tune set from train_set data
gc()

# We are changing the name of VALIDATION_DF TO VAL_SET.
val_set <- append_user_movie_avgs_preds(train_set, validation_df) # Predict validation set from train_set data.
cat('val_set nrows: ', nrow(val_set), "\n")
gc()

# RMSE using just the AVERAGE RATING ACROSS ALL REVIEWS
# RMSE calculated using TRAIN_SET model parameters, on TEST_SET to predict test_set ratings.
test_rmses <- c()
test_rmses["average"] <- recommenderlab::RMSE(test_set$mean_rating, test_set$rating) 

## CALCULATING RMSE FOR VALIDATION SET USING THIS STRATEGY
test_rmses["movieonly"] <- recommenderlab::RMSE(test_set$pred_movieonly, test_set$rating)
test_rmses["useronly"] <- recommenderlab::RMSE(test_set$pred_useronly, test_set$rating)
test_rmses["movieanduser"] <- recommenderlab::RMSE(test_set$pred_both, test_set$rating)

cat(names(test_rmses), "\n")
cat(test_rmses, "\n")
# We see that our RMSEs line up well compared to what we saw in Course 8.
# rm(test_set)
gc()



##########################################################
### SECTION 7: USING GENRES AS PREDICTORS (OUR METHOD) ###
##########################################################


# See how many unique users are in our datasets
unique_train_set_userId <- train_set %>% distinct(userId)
unique_val_set_userId <- validation_df %>% distinct(userId)
unique_edx_set_userId <- edx_df %>% distinct(userId)
nrow(unique_train_set_userId)
nrow(unique_val_set_userId)
nrow(unique_edx_set_userId)

# Create matrix of userIds vs. genre preference weights (nrow(userId) x 20 genres)
user_genre_mat <- matrix(0, nrow(unique_edx_set_userId), length(unique_genres))
rownames(user_genre_mat) <- unique_edx_set_userId$userId #Rows are named by userId
colnames(user_genre_mat) <- usergenre_binary_header_idx #Cols are named by genre.
user_genre_mat <- as_data_frame(user_genre_mat)
user_genre_mat <- user_genre_mat %>% mutate(userId = unique_edx_set_userId$userId)

# Now, we have to begin filling up this matrix of user genre preferences. 
# Procedure:
# 1. Look at a given rating observation. 
# 2. Look at that user's bias and movie's bias (if avaliable, if not assume = 0)
# 3. Get a rating predicted by the user and movie biases.
# 4. If this rating is lower than the average of the user's ratings, negatively weight the relevant genres.
#    If this rating is higher than [...], positively weight the relevant genres.
# 5. Do this for all rating observations, getting a "genre preference matrix" filled out.

# New model assumes that the remaining deviation (after accounting for user and movie bias) is due to user genre preference.

# We will be using the train set.
# We need:
# - mean rating from train set
# - mean user rating, and bias relative to mean rating
# - mean movie rating, and bias relative to mean rating
# - actual rating in train set
# We need to calculate:
# - difference between true rating and pred_both
# - the observation's modified genre vector
# Then add modified genre vector to the userId's genre vector in the matrix user_genre_mat
# Do this for every observation. 

mean_train_rating <- mean(train_set$rating)
mean_train_rating 

# Define a tunable hyperparameter, lambda, for weighting how much significance we give the genres.
# This will be used after the genre matrix is generated, when we're calculating our predictions on val set.
lambda <- 0.1

# Remember, all of the prediction calcs use the train ratings ONLY.
# The ratings in test, tune and val are used ONLY for calculating error, NOT predictions.

# Preallocate dataframe for individual user ratings' genre affinities
user_genre_df_train_individual <- data.frame(matrix(nrow=nrow(train_set), ncol=length(unique_genres)+1))
colnames(user_genre_df_train_individual) <- c("userId", usergenre_binary_header_idx)

user_genre_df_train_individual[, usergenre_binary_header_idx] <- train_set[, genre_binary_header_idx]*(train_set[,"pred_both"]-train_set[, "rating"])*-1
user_genre_df_train_individual[, "userId"] <- train_set$userId
head(user_genre_df_train_individual)

# Collapse individual rating observations into a user_genre_matrix form. 
# From 1 row per observation (multiple rows per user) to 1 row per user, averaging out the g_x columns per user.
user_genre_df_train_collapsed <- user_genre_df_train_individual %>% group_by(userId) %>% summarize_at(vars(usergenre_binary_header_idx), mean)
rm(user_genre_df_train_individual)
gc()

head(user_genre_df_train_collapsed)
# This shows the head of the user genre preferences. Negative means user dislikes that genre, positive is likes genre.


### TRAIN SET ASSIGNMENT ###

# Now that we have a matrix of users' genre preferences, we can apply this to predict ratings.
train_set <- left_join(train_set, user_genre_df_train_collapsed, by="userId")
head(train_set,2)
gc()

# Generate element-wise multiplication of g_x and ug_x vectors per row.
# Sum these up per row, store in a column called: genre_weight_sum
train_set[, "genre_weight_sum"] <- rowSums(train_set[, genre_binary_header_idx] * train_set[, usergenre_binary_header_idx])
head(train_set, 2)

### TUNE SET ASSIGNMENT ###
tune_set <- left_join(tune_set, user_genre_df_train_collapsed, by="userId")
tune_set[, "genre_weight_sum"] <- rowSums(tune_set[, genre_binary_header_idx] * tune_set[, usergenre_binary_header_idx])
head(tune_set, 2)

### TEST SET ASSIGNMENT ###
test_set <- left_join(test_set, user_genre_df_train_collapsed, by="userId")
test_set[, "genre_weight_sum"] <- rowSums(test_set[, genre_binary_header_idx] * test_set[, usergenre_binary_header_idx])
head(test_set, 2)

### VAL SET ASSIGNMENT ###
val_set <- left_join(val_set, user_genre_df_train_collapsed, by="userId")
val_set[, "genre_weight_sum"] <- rowSums(val_set[, genre_binary_header_idx] * val_set[, usergenre_binary_header_idx])
head(val_set, 2)

gc()
# We are now ready to tune lambda! Next block


##################################################
### SECTION 8: MODEL EVALUATION AND COMPARISON ###
##################################################

temp_rmse_train <- matrix(0, length(seq(0,3,0.1)), 2) # Matrix to store lambdas / RMSEs in.

counter_1 <- 1
for (lambda_i in seq(0, 3, 0.1)) {
  temp_rmse_train[counter_1, 1] <- recommenderlab::RMSE(train_set$pred_both + lambda_i*train_set$genre_weight_sum, train_set$rating)
  temp_rmse_train[counter_1, 2] <- lambda_i
  cat("RMSE with lambda_i TRAIN = ", lambda_i, " is ", temp_rmse_train[counter_1,1], "\n")
  counter_1 <- counter_1 + 1
}
test_rmses["movieanduser"]

# We see lambda_i around 1.6 minimizes RMSE for train. Lets tune this again using the tune set.
temp_rmse_tune <- matrix(0, length(seq(0,3,0.1)), 2) # Matrix to store lambdas / RMSEs in.
counter_2 <- 1
for (lambda_i in seq(0, 3, 0.1)) {
  temp_rmse_tune[counter_2,1] <- recommenderlab::RMSE(tune_set$pred_both + lambda_i*tune_set$genre_weight_sum, tune_set$rating)
  temp_rmse_tune[counter_2,2] <- lambda_i
  cat("RMSE with lambda_i TUNE = ", lambda_i, " is ", temp_rmse_tune[counter_2,1], "\n")
  counter_2 <- counter_2 + 1
}
test_rmses["movieanduser"]

# Select lowest RMSE lambda. 
lambda <- temp_rmse_tune[which.min(temp_rmse_tune[,1]), 2]

test_rmses["movieusergenre"] <- recommenderlab::RMSE(test_set$pred_both + lambda*test_set$genre_weight_sum, test_set$rating)
test_rmses

test_set <- test_set %>% mutate(genre_pred_rating = pred_both + lambda*genre_weight_sum)
head(test_set, 2)

# RMSE is around 0.86! But we can do better by limiting ratings that fall outside  0 to 5.
head(test_set[which(test_set$genre_pred_rating > 5 | test_set$genre_pred_rating < 0),])
test_set[which(test_set$genre_pred_rating > 5), "genre_pred_rating"] <- 5
test_set[which(test_set$genre_pred_rating < 0), "genre_pred_rating"] <- 0
test_rmses["movieusergenre_edgelimited"] <- recommenderlab::RMSE(test_set$genre_pred_rating, test_set$rating)
test_rmses

## APPLY LAMBDA = 1.3 to final model in VALIDATION DATA
val_set <- val_set %>% mutate(genre_pred_rating = pred_both + lambda*genre_weight_sum)
# Limit ratings by cutting them off from 0 to 5.
val_set[which(val_set$genre_pred_rating > 5), "genre_pred_rating"] <- 5
val_set[which(val_set$genre_pred_rating < 0), "genre_pred_rating"] <- 0
# Calculate final RMSE
val_rmse_final <- recommenderlab::RMSE(val_set$genre_pred_rating, val_set$rating)
val_rmse_final

# Predicted ratings for val_set are in genre_pred_rating.











