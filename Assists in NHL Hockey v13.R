## ----setup, include=FALSE---------------------------------------------------------------------------------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)
knitr::opts_chunk$set(message = FALSE)



## ----Read NHL Player Data---------------------------------------------------------------------------------------------------------------------------

# Install packages as required.
# To handle data piping, mutations, etc.
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
# For classification and regression tree models.
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
# Data tables at download.
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
# Tools that help with aggregations and data handling.
if(!require(cgwtools)) install.packages("cgwtools", repos = "http://cran.us.r-project.org")
# To display pretty tables.
if(!require(kableExtra)) install.packages("kableExtra", repos = "http://cran.us.r-project.org")
# Random forest models.
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
# Regression tree models (and classification trees, but not used here).
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
# To support plotting of regression trees.
if(!require(rpart.plot)) install.packages("rpart.plot", repos = "http://cran.us.r-project.org")
if(!require(party)) install.packages("party", repos = "http://cran.us.r-project.org")
# For pretty plots of regression trees.
if(!require(partykit)) install.packages("partykit", repos = "http://cran.us.r-project.org")
# To plot correlations.
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
# Invoke libraries.
library(corrplot)
library(partykit)
library(rpart)
library(rpart.plot)
library(tidyverse)
library(caret)
library(data.table)
library(kableExtra)
library(cgwtools)
library(rpart)
library(randomForest)






## ----General Data Exploration, cache=TRUE-----------------------------------------------------------------------------------------------------------
# This code automatically downloads the required files from a publicly 
#  shared Google Drive location.
#  There is a commented block at the end of this section for backup
#  if there are problems with the automatic download.

dl_game_skater_stats <- tempfile()
download.file("https://drive.google.com/uc?id=1M7MZS_vTs3rWu7R_c0MbI0Vdp7vQwdtf", dl_game_skater_stats)
game_skater_stats <- read.csv(dl_game_skater_stats)
head(game_skater_stats,3)

dl_player <- tempfile()
download.file("https://drive.google.com/uc?id=1wNzxzlk6n104mTWIN4Em8rf8wW_xJw8K", dl_player)
player_info <- read.csv(dl_player)
head(player_info,3)

dl_game <- tempfile()
download.file("https://drive.google.com/uc?id=13YxSU7uTtVaRP1pBYVP8DzSA60tEy9kQ", dl_game)
game <- read.csv(dl_game)
head(game,3)

dl_team <- tempfile()
download.file("https://drive.google.com/uc?id=1qJ1XWY072JmQQTkTLgIqCHA6niAvQ_Le", dl_team)
team_info <- read.csv(dl_team)
head(team_info,3)

# Only uncomment and use this code block if automatic download fails.
#  The CSV files should be downloaded manually from here:
#  https://github.com/henryawong/edX
#  Place them in R's before running the code below.
#  
#  ### Code block starts here ###
#  game_skater_stats  <- read.csv("game_skater_stats.csv")
#  player_info  <- read.csv("player_info.csv")
#  team_info  <- read.csv("team_info.csv")
#  game <- read.csv("game.csv")
#  ### Code bock ends here ###



## ----Skater Data------------------------------------------------------------------------------------------------------------------------------------

# Join skater data with tables for player attributes,
#  game information, and team information.
player_game_data <- game_skater_stats %>%
  left_join(player_info, by="player_id") %>%
  left_join(game, by="game_id") %>%
  left_join(team_info, by="team_id") %>%
#  Add a points field.  
  mutate(points = goals + assists) %>% 
#  Eliminate any duplicate rows (from joins or originals)
  unique() %>%
#  Narrow to only wanted fields.  
#  Keep other fields in mind for future study.
  select(firstName, lastName, season
       , timeOnIce, assists, goals, points, shots, hits
       , season, game_id, player_id
       , teamName, team_id
       , height, weight, birthDate, nationality, primaryPosition
       , date_time_GMT)

# Sample rows.
head(player_game_data,3)



## ----Count Skater Rows------------------------------------------------------------------------------------------------------------------------------
# Show a data sample.
player_game_data %>% head(3)

# Count player game rows. 
nrow(player_game_data)

# Count players. View as factor and count unique levels.
factor(player_game_data$player_id) %>% 
  levels() %>% length()

# Count teams.  Use levels function to count as factor.
factor(player_game_data$team_id) %>% 
  levels() %>% length()

# Count seasons.
factor(player_game_data$season) %>% 
  levels() %>% length()

# List seasons, sorted.  Note missing 2004/2005 canceled season.
factor(player_game_data$season) %>% 
  levels() %>% sort() %>% data.frame()

# Games counts per season.
names(game)
game %>% group_by(season) %>% 
  select(game_id) %>% unique() %>% 
  summarize(gameCount=n())


## ----Season Summary 1, message=FALSE----------------------------------------------------------------------------------------------------------------

# Create a data summary of 
#  league assists grouped by season.
#  Summarize by computing sums for goals, points, and assists.
season_summary <- player_game_data %>% group_by(season) %>% 
  summarize(sumAssists=sum(assists)
            , sumGoals=sum(goals)
            , sumPoints=sum(points))

# Turn off scientific notation to keep axis labels 
#  clean in the following plots and graphs.
options(scipen=999)

# Graph total league assists by included seasons. 
#  Make markers large and color them red.
#  Turn the season labels on the x-axis to 90 degrees to avoid crowding.
#  Eliminate the legend.
#  Label with a title and axis labels.
#  Put colored horizontal lines at y values of 13000 and 11500 to
#  help the eye spot ranges.
season_summary %>% 
  ggplot(aes(x=as.factor(season), y=sumAssists)) +
          geom_point(color="red", size=4) +
          theme(axis.text.x=element_text(angle=90) ,legend.position = "none") +
          labs(title = "League Assists by Season", y="Total Assists", x="Season") +
          geom_hline(yintercept=13000, colour="blue") +
          geom_hline(yintercept=11500, colour="darkgreen") 




## ----Season Summary Averages------------------------------------------------------------------------------------------------------------------------
# Mean of all league season assist totals.
mean(season_summary$sumAssists)

# Remove 20122013 and recalculate mean.
season_summary %>% filter(season != 20122013) %>% 
  summarize(avgSumAssists = mean(sumAssists))

# Remove lower cluster and outlier year and recalculate.
season_summary %>% 
  filter(season != 20122013) %>% 
  filter(season > 20052006) %>% 
  summarize(avgSumAssists = mean(sumAssists))



## ----Player Seasons---------------------------------------------------------------------------------------------------------------------------------
# Prepare data to graph season assist totals per player.
#  Must group by season, then player within a season.  
#  Compute total goals, assists and points.
player_season_summary <- player_game_data %>% group_by(season, player_id) %>%
      summarize(sumAssists=sum(assists)
            , sumGoals=sum(goals)
            , sumPoints=sum(points))




## ----Graph Player Season Assists by Player----------------------------------------------------------------------------------------------------------
# Graph season assists by player.
#  Blank out player IDs on the x-axis, as they are meaningless here.
#  Make points blue and dots small.
player_season_summary %>%  ggplot(aes(x=player_id, y=sumAssists)) +
          geom_point(color="blue", size=1) +
          theme(axis.text.x=element_blank(),legend.position = "none") +
          labs(title = "Season Assists by Player", y="Total Assists", x="Player") 
          

# Minimum, maximum, and interquartile range.
summary(player_season_summary) 

# Calculate standard deviation. 
sd(player_season_summary$sumAssists) 



## ----Point Players----------------------------------------------------------------------------------------------------------------------------------


#  Filter to only games where players produce points.
players_with_points <- player_game_data %>%
  filter(points > 0) 

# Graph goals versus assists per player for games where the player 
#  earns points.  Use a scatterplot.
#  Goals on the x-axis, assists on the y-axis, points as dot size.
#  Assists in brighter shading.
#  Put markers on every integer on the axes.
#  Create in a graph object for layers added later.
plot_points <- players_with_points %>%
  ggplot(aes(goals, assists, color=assists)) +
  geom_jitter() +
  scale_x_continuous(breaks = seq(1, 5,1)) +
  scale_y_continuous(breaks = seq(1, 5,1))

# Show graph by calling graph object.
plot_points




## ----Player Point Correlations----------------------------------------------------------------------------------------------------------------------

# Calculate correlation between goals and assists over all player games.
cor(player_game_data$assists, player_game_data$goals)

# Correlate them in games where a player earns at least 1 goal or 1 assist.
cor(players_with_points$assists, players_with_points$goals)



## ----Goals vs Assists by Position-------------------------------------------------------------------------------------------------------------------

# Graph points per player per game.
#  Add grouping by player's ice position.
#  Emphasize GOALS with color fading in high contrast from YELLOW to BLACK.
#  Label it clearly as Goal Share of Points.
players_with_points %>% group_by(primaryPosition) %>%
  ggplot(aes(x=primaryPosition, y=points, color=goals)) +
    geom_jitter() +
    scale_color_gradient(low = 'yellow', high = 'black') +
    labs(title = "Goal Share of Player Game Points by Position", x="Primary Position", y="Points")

# Graph points per player per game.
#  Add grouping by player's ice position.
#  Emphasize ASSISTS with color fading in high contrast from GRAY to PURPLE.
#  Label it clearly as Goal Share of Points.
players_with_points %>% group_by(primaryPosition) %>%
  ggplot(aes(x=primaryPosition, y=points, color=assists)) +
    geom_jitter() +
    scale_color_gradient(low = 'gray', high = 'purple') +
    labs(title = "Assist Share of Player Game Points by Position", x="Primary Position", y="Points")



## ----Graph Assist Games Count by Age----------------------------------------------------------------------------------------------------------------
# Make a variable for analyzing player age versus assists.
#  Calculate field for age as game day minus player birthday (results in days).
#  Convert the days result to years (leap years are not specially treated here). 
player_game_age_assists <- players_with_points %>% 
  mutate(ageAtGameDays = as.Date(date_time_GMT)-as.Date(birthDate)) %>%
  mutate(age = as.numeric(round(ageAtGameDays/365))) %>% select(age, assists)

# Since we started with players with POINTS, filter to only assists.
#  Graph games with 1+ assist counts in a histogram with default bins.
#  Add some color and outline to make it readable.
#  Mark every age on the y-axis, from 18 to 47 on the x-axis.
#  Add labels.
player_game_age_assists %>% filter(assists>0) %>%
    ggplot(aes(x=age)) +
      geom_histogram(col="blue", fill="brown") +
      scale_x_continuous(breaks = seq(18,47,1)) +
      labs(title="Games with 1+ Player Assists by Player Age"
           , y="Games with 1+ Assists"
           , x="Player Age on Game Day")


## ----Age vs Assists Graph 1-------------------------------------------------------------------------------------------------------------------------

# Since we started with players with POINTS, filter to only assists.
#  Graph age by game assists counts.
#  Use a point plot.  
#  Mark every age on the y-axis, from 18 to 47 
#  (using a 1-step sequence) with a tick.
#  Add a dot size for count of games.
#  Add labels.
player_game_age_assists %>% filter(assists>0) %>%
  group_by(age) %>% ggplot(aes(y=age, x=assists)) +
    geom_point() +
    scale_y_continuous(breaks = seq(18,47,1)) +
    geom_count() +
    labs(title = "Age by Assists Per Game", x="Assists", y="Player Age")




## ----Shots------------------------------------------------------------------------------------------------------------------------------------------

# Create a variable summing assists and shots by season by player.
shots_assists <- player_game_data %>% group_by(season, player_id) %>%
  summarize(sumAssists=sum(assists), sumShots=sum(shots), sumGoals=sum(goals)) 

# Plot player season assists vs player season shots.
#  Graph as scatter (jitter) in color.
#  Add a smoothing line with defaults.
#  Change marks on axes to more often than the default.
#  Start, end, and step for break sequences.
#  Add labels and title.
shots_assists %>%
    ggplot(aes(y=sumAssists, x=sumShots)) + 
    geom_jitter(colour="blue") +
    geom_smooth(color="red") +
    scale_y_continuous(breaks = seq(0,100,10)) +
    scale_x_continuous(breaks = seq(0,600,50)) +
    labs(title="Player Season Assists by Player Season Shots"
         ,y="Season Player Assists", x="Season Player Shots")




## ----Shot Correlation-------------------------------------------------------------------------------------------------------------------------------

# Create a correlation matrix to plot.
cor_all <- shots_assists %>% select(sumAssists, sumShots, sumGoals) %>% 
  data.frame() %>% cor()

# Show correlations in a table.
cor_all  %>% round(3) %>% kable()

# Plot it graphically
corrplot(cor_all, method = "color", type="upper")



## ----TOI--------------------------------------------------------------------------------------------------------------------------------------------

# First, calculate sum of assists and 
#  mean and sum of time on ice by season and player.
toi_assists <- player_game_data %>% group_by(season, player_id) %>%
  summarize(sumAssists=sum(assists), meanTOI=mean(timeOnIce)
            ,sumTOI=sum(timeOnIce)) 

# Sample rows.
head(toi_assists,3)

# Graph assists versus time on ice aggregates.
#  Use a scatter (jitter) plot with some color.
#  Add a smoothing line to show trending.
#  Change axis ticks.
#  Add vertical lines at 1160 and 1310 meanTOI.
#  to show flat segment. 
#  Add labels.
toi_assists %>%
    ggplot(aes(y=sumAssists, x=meanTOI)) + 
    geom_jitter(colour="red") +
    geom_smooth(colour="black") + 
    scale_y_continuous(breaks = seq(0,120,10)) +
    scale_x_continuous(breaks = seq(0,2000,100)) +
    geom_vline(xintercept = 1160, color="blue"
               , linetype="dashed", size=1) +
    geom_vline(xintercept = 1310, color="blue"
               , linetype="dashed", size=1) +
    labs(title="Season Player Assists by Mean Time On Ice"
         , y="Player Assists",x="Average Season Ice Time (Minutes)")




## ----Correlate Ice Time-----------------------------------------------------------------------------------------------------------------------------
# Calculate correlation.
#  Correlation is high so shots are redundant with sumAssists as a predictor.
cor(toi_assists$meanTOI, toi_assists$sumAssists)




## ----Nationality------------------------------------------------------------------------------------------------------------------------------------

# Count the unique players in the database.  Bear in mind, players
#  come and go over seasons, but this is fairly representative.
unique_player_count <- player_game_data %>% 
  select(player_id) %>% unique() %>% nrow() 

# Show the database player count.
unique_player_count

# Group game data by nationality, then player_id.
#  Take the unique combinations (before this step, each row is a player game).
#  Calculate counts and percentages.
player_game_data %>%  select(nationality, player_id) %>% 
  unique() %>% group_by(nationality) %>% 
  summarize(playerCount=n()
            , natPercent=playerCount/unique_player_count) %>%
              arrange(desc(playerCount))



## ----Nationality Graph Season Sum Assists, message=FALSE--------------------------------------------------------------------------------------------

# Aggregate player game stats by nationality.
#  Filter to only those games where player assists occur.
#  Group by season, then nationality and calculate aggregates.
agg_nationality <- player_game_data %>% 
  filter(assists > 0) %>%
  group_by(season, nationality) %>%
  summarize(nation=nationality, gameCount=n(), gamePoints = sum(points)
            , gameAssists = sum(assists), gameGoals = sum(goals)
            , meanPoints = mean(points), meanAssists = mean(assists)
            , meanGoals = mean(goals)) %>%
                unique()

# Sum seasons assists by nationality.
#  Compute average of these seasonal assists sums.
#  Show the top 7.
player_game_data %>% 
  select(nationality, season, assists) %>%
  group_by(season, nationality) %>%
    summarize(sumSeasonAssists=sum(assists)) %>%
  group_by(nationality) %>%
    summarize(meanSeasonAssists=mean(sumSeasonAssists)) %>%
    arrange(desc(meanSeasonAssists)) %>% head(7)

# Plot seasonal assist totals by player nationality.
#  Add a scatter plot geom.
#  Show colors to differentiate seasons but hide legend.
#  (Season isn't the focus of this graph).
#  Add title and labels, bearing axis flip in mind.
#  Flip the axes to show nationalities as a readable vertical list.
agg_nationality %>%
  ggplot(aes(x=nationality, y=gameAssists)) +
  geom_jitter(aes(color=season), show.legend = FALSE) +
    scale_colour_gradient(low="blue",high="green") +
    labs(title = "Season Assists by Player Nationality"
       , x="Nationality", y="Season Assists") +
      coord_flip()





## ----Nationality Graph Season Mean Assists----------------------------------------------------------------------------------------------------------
# Plot mean season game assists by nationality.
#  On a scatter plot, show game count by size, season by color.
#  Hide legend as specific seasons isn't the focus.
agg_nationality %>% 
  ggplot(aes(x=nation, y=meanAssists)) +
  geom_jitter(aes(size = gameCount, color=season), show.legend = FALSE) +
  labs(title = "Mean Game Assists in Player 1+ Assist Games by Player Nationality"
       , x="Nationality", y="Average Game Assists") +
      coord_flip()


# Compute personal averages for Kopitar and McDavid.
#  Filter by the two players (Kopitar has played more seasons).
#  Group in order of seasons, then by the players' names.
#  Sum by this grouping to get seasonal player totals.
#  Regroup by only the player names and calculate mean of
#  their season sum of assists.
player_game_data %>% 
  filter(lastName=="Kopitar" | lastName=="McDavid") %>%
  select(firstName, lastName, season, assists) %>%
  group_by(season, firstName, lastName) %>%
  summarize(sumAssists=sum(assists)) %>%
  group_by(firstName, lastName) %>%
  summarize(meanSeasonAssists=mean(sumAssists))

# Us the same logic as above, without filters, to
#  show the highest means season assist players.
player_game_data %>% 
  select(firstName, lastName, season, assists, nationality) %>%
  group_by(season, firstName, lastName, nationality) %>%
  summarize(sumAssists=sum(assists)) %>%
  group_by(firstName, lastName, nationality) %>%
  summarize(meanSeasonAssists=mean(sumAssists)) %>%
  arrange(desc(meanSeasonAssists)) %>% head(10)





## ----Team Seasons-----------------------------------------------------------------------------------------------------------------------------------

# Group player game data by season and team identifiers
#  Calculate sums and averages for assists, goals and hits.
team_seasons <- player_game_data %>%
  group_by(season, team_id, teamName) %>%
  unique() %>% summarize(seasonAssists=sum(assists)
                         ,seasonGoals=sum(goals)
                         ,seasonHits=sum(hits)
                         ,seasonAvgAssists=mean(assists)
                         ,seasonAvgGoals=mean(goals)
                         ,seasonAvgHits=mean(hits)) %>%
                          ungroup()
  

# Sort by highest season assists by team. Show top 10.
team_seasons %>%
  arrange(desc(seasonAssists)) %>%
  head(10) %>% select(season, teamName, seasonAssists)


# Sort by highest season assists by team. Show top 10.
team_seasons %>% 
  arrange(desc(seasonGoals)) %>%
  head(10) %>% select(season, teamName, seasonGoals)


# Sort by highest season assists by team. Show top 10.
team_seasons %>% 
  arrange(desc(seasonHits)) %>%
  head(10) %>% select(season, teamName, seasonHits)





## ----Team Players-----------------------------------------------------------------------------------------------------------------------------------

# Compute average season team assists.
#  Eliminate null records from previous grouping operations.
ta <- team_seasons %>% select(season, teamName, seasonAvgAssists) %>% 
  filter(!is.na(teamName)) %>%
  arrange(desc(seasonAvgAssists)) 

# Summarize assists.
ta$seasonAvgAssists %>% summary()

# Standard deviation with season separation.
ta$seasonAvgAssists %>% sd()

# Graph season average assists by team
ta %>% ggplot(aes(x=season, y=seasonAvgAssists, color=teamName)) +
       geom_point() +
       theme(axis.text.x = element_blank()) +
       labs(y="Season Team Average Assists"
            , x="Seasons"
            , title = "Average Assists by Season")

# Average team assists over 19 seasons.
all_team_assists <-player_game_data %>% 
  group_by(teamName) %>% 
  summarize(meanTeamAssists=mean(assists)) %>%
  arrange(desc(meanTeamAssists)) %>%
  filter(!is.na(teamName))

# List team assists over 19 seasons.
all_team_assists %>% arrange(desc(meanTeamAssists))

# Distribution of assists over 19 seasons. 
all_team_assists$meanTeamAssists %>% summary()

# SD of assists,
all_team_assists$meanTeamAssists %>% sd()



## ----Defease Team Variables from Core Data----------------------------------------------------------------------------------------------------------

# Delete team identifiers from core data.
player_game_data <- player_game_data %>% select(-team_id, -teamName)





## ----Correlations 1---------------------------------------------------------------------------------------------------------------------------------

# Create a subset of the player game data to correlate.  
#  Make it a tibble.
cc <- player_game_data %>% 
  select(timeOnIce, assists, goals, shots, hits, points) %>% 
  as_tibble()

# Create an object grouped by season, then player.
#  Aggregate the stats to correlate.  
cd <- player_game_data %>% group_by(season, player_id) %>%
  summarize(  pointGamesCount=n(), seasonPoints = sum(points)
            , seasonAssists = sum(assists), seasonGoals = sum(goals)
            , seasonTOI = sum(timeOnIce), seasonShots=sum(shots) )%>%
              unique()

# Create a correlation matrix for all fields
cm <- cor(cd[,-c(1,2,3)])
cm

# Plot correlation matrix.
corrplot(cm, method = "number", type = "upper")



## ----Data Grain Prep--------------------------------------------------------------------------------------------------------------------------------

# Select the fields we need for models.
#  Group first by season, then player.
#  Total and average based on this grouping hierarchy.
season_data <- player_game_data %>% 
  select(player_id, firstName, lastName, season
                             , game_id, nationality
                             , points, goals, assists, shots
                             , hits, timeOnIce) %>% 
  group_by(season, player_id) %>% summarise(sumPoints=sum(points)
                                            , sumAssists=sum(assists)
                                            , sumGoals=sum(goals)
                                            , sumShots=sum(shots)
                                            , sumTimeOnIce=sum(timeOnIce)
                                            , meanAssists=mean(assists)
                                            , meanGoals=mean(goals)
                                            , meanShots=mean(shots)
                                            , meanTimeOnIce=mean(timeOnIce))
# Show sample rows.  
season_data %>% arrange(desc(sumPoints)) %>% head(4)


## ----Stage Data-------------------------------------------------------------------------------------------------------------------------------------

# Prepare a "next season" data set.
#  Create a join_season field to avoid confusion.
#  This will represent the "prior season" in the join at a later step.
#  Only for season 20052006, the prior season is forced to
#  20032004 because 20042005 was canceled.
#  Rename season field to next_season because that's what it
#  will become once joined back to the main data, below.
#  Select only the needed fields to avoid duplication in the join.
next_season_data <- season_data %>% 
  mutate(join_season=ifelse(season==20052006, 20032004, season-10001)) %>%
  rename(next_season=season, nextSumAssists=sumAssists) %>%
  select(join_season, next_season, player_id, nextSumAssists)

# Preview some rows.
next_season_data %>% head(3)

# Join main data back to next season data.
agg_data <- season_data %>% 
  full_join(next_season_data, by=c("season"="join_season", "player_id"="player_id"))

# Display relevant rows to demonstrate prediction data structure.
#  Note season versus next season.
agg_data %>% arrange(desc(sumAssists)) %>% 
  select(player_id, season, sumAssists, next_season, nextSumAssists)

# Filter out records with null next_seasons.  They have no
#  values for next season assist totals to  
#  compare predictions against.  These will be 20192020 stats.
agg_data <- agg_data %>% filter(!is.na(next_season)) 

# Unique player count in new dataset.
agg_data$player_id %>% unique %>% length()

# Confirm that unique player count has not changed from the 
#  original data set.
agg_data$player_id %>%
  unique %>% 
  length() == player_game_data$player_id %>% 
               unique() %>% 
               length()

# Because of the construction method of these data sets, the year
#  before a rookie joins will show the previous year (when he did not play)
#  with a "next year" nextSumAssists value for him but no "current year"
#  sumAssists or any other predictors.  Remove these records.  
#  Filter out rows with null current year predictors generated by data
#  handling.  These nulls were not in the original set.
agg_data <- agg_data %>% filter(!is.na(sumAssists))
                     
# Calculate ending row count.
nrow(agg_data)

                     



## ----Set Up Experiment Data, message=FALSE----------------------------------------------------------------------------------------------------------

# Establish range covered by data set
# 2000 to 2019.
agg_data$season %>% unique()

# Set 20172018 to 20192020 as validation
validation <- agg_data %>% 
  filter(season==20172018 | season==20182019 | season==20192020)

# Use 20142015 to 20162017 as test.  
test_set <- agg_data %>% 
  filter(season==20142015 | season==20152016 | season==20162017) 

# The rest is for training.
train_set <- agg_data %>% 
  anti_join(test_set) %>% 
  anti_join(validation) 

# Clear NA records generated by previous grouping.
train_set <- train_set %>% filter(!is.na(sumAssists))
test_set <- test_set %>% filter(!is.na(sumAssists))
validation <- validation %>% filter(!is.na(sumAssists))

# There will be some rookies in the test set years and validation set 
#  years that will not be in the training years.  Unfortunately, because
#  of time linearity, we must remove them because prediction of rookie
#  stats with no history is tough for our models, which are intended to
#  predict season assists from using historical data for known players.
# 
# If we put the rookie or retired player data into the training set the seasonal
#  aggregates for certain groupings will be disturbed, so we omit them altogether.
#  Execute on kept validation rows.
validation <- validation %>% semi_join(train_set, by = "player_id")

# Do the same for the training set.
test_set <- test_set %>% semi_join(train_set, by = "player_id") 

# Show sample rows in training set.
train_set %>% head(3)

# Rowcounts
c(nrow(train_set), nrow(test_set), nrow(validation))



## ----Compute SD of Season Player Assists------------------------------------------------------------------------------------------------------------


# Summarize player season stats.
player_season_summary$sumAssists %>% summary()

# Calculate standard deviation of all player seasons. 
sd(player_season_summary$sumAssists) 





## ----RMSE Function----------------------------------------------------------------------------------------------------------------------------------
# Build the RMSE function to evaluate models.
RMSE <- function(actual_season_assists, predicted_season_assists){
  sqrt(mean((actual_season_assists - predicted_season_assists)^2))
}


## ----RT Model 1-------------------------------------------------------------------------------------------------------------------------------------
# Train regression tree model 1 using ice time, shots, and assists.
#  Game level predictors for season stats.
fit_rt1 <- rpart(nextSumAssists~meanTimeOnIce + sumShots + sumAssists 
                , method="anova", data=train_set)




## ----Print R1 Fit-----------------------------------------------------------------------------------------------------------------------------------

# Print decision rules, two different ways.
print(fit_rt1)
rpart.rules(fit_rt1)

# Show tree logic.
rpart.plot(fit_rt1)




## ----More RT1---------------------------------------------------------------------------------------------------------------------------------------

# Generate predictions against test set.
predictions_rt1 <- predict(fit_rt1, newdata=test_set)

# Show possible prediction outputs.
unique(predictions_rt1) %>% head()

# Calculate RMSE.  Store for summary at end.
(rmse_rt1 <- RMSE(test_set$nextSumAssists, predictions_rt1))



## ----Variable Importance RT1------------------------------------------------------------------------------------------------------------------------

# Show variable importance in descending rank.
varImp(fit_rt1) %>% arrange(desc(Overall))



## ----RT1 Summary------------------------------------------------------------------------------------------------------------------------------------

# Complexity parameter cross-validation.
summary(fit_rt1) 


## ----R Sqaure RT1-----------------------------------------------------------------------------------------------------------------------------------
# Plot R-square by number of splits.
rsq.rpart(fit_rt1)   



## ----RT Model 2-------------------------------------------------------------------------------------------------------------------------------------
# Train RT model 2
fit_rt2 <- rpart(nextSumAssists~. , method="anova", data=train_set)

# Plot decision rules.
rpart.plot(fit_rt2)

# Rank variable importance.
varImp(fit_rt2) %>% arrange(desc(Overall))


## ----More RT2---------------------------------------------------------------------------------------------------------------------------------------
# Predict using testing set.
predictions <- predict(fit_rt2, newdata=test_set)

# Calculate and store RMSE.
(rmse_rt2 <- RMSE(test_set$nextSumAssists, predictions))

# List possible predictions.
predictions %>% unique()

# Compute distribution of possible player season assists.
train_set$nextSumAssists %>% summary()  



## ----More RT2 Detail--------------------------------------------------------------------------------------------------------------------------------

# Print complexity parameter analysis.  
printcp(fit_rt2) 

# Plot R-sqare.
rsq.rpart(fit_rt2)



## ----Linear Model 1, message=FALSE, warning=FALSE---------------------------------------------------------------------------------------------------
# Train LM model 1
fit_lm1 <- lm(nextSumAssists ~ sumGoals + sumAssists, data=train_set)

# Use the model to predict.
pred <- predict(fit_lm1, newdata=test_set)

# Compute RMSE.
(rmse_lm1 <- RMSE(test_set$nextSumAssists, pred))

# Summarize model.
summary(fit_lm1)




## ----Linear 1a, warning=FALSE-----------------------------------------------------------------------------------------------------------------------
# Tain a model dropping the player_id predictor.
fit_lm1a <- lm(nextSumAssists ~ . -player_id, data=train_set)
predictions_lm1a <- predict(fit_lm1a, newdata=test_set)

# Compute new RMSE.  Not much different.
(rmse_lm1a <- RMSE(test_set$nextSumAssists, predictions_lm1a))

# Show model details.  
fit_lm1a

# Show linear model details
summary(fit_lm1a)



## ----Ensemble Model RT 3 Step, warning=FALSE--------------------------------------------------------------------------------------------------------

# Run predictions using linear model with exhaustive variables.
pred_lm <- predict(fit_lm1a, newdata = train_set)

# Add pred_lm to train set.
train_set_lm <- cbind(train_set, pred_lm)

# Rename added field.
train_set_lm <- train_set_lm %>% 
  rename(lm_nextSumAssists="...14")

# See sample rows.
train_set_lm %>% head(3)

# Add pred_lm to test set.
pred <- predict(fit_lm1a, newdata=test_set)
test_set_lm <- cbind(test_set, pred)

# Rename added field.
test_set_lm <- test_set_lm %>% 
  rename(lm_nextSumAssists="...14")

# See relevant rows in sample rows.
train_set_lm %>% 
  select(player_id, sumAssists, nextSumAssists, lm_nextSumAssists) %>% 
  head(3)

# Build regression tree model using linear predictions as a feature.
fit_lm1_rt3 <- rpart(nextSumAssists~., method="anova", data = train_set_lm)

# Make predictions using this new model.
predictions_ensemble <- predict(fit_lm1_rt3, newdata = test_set_lm)

# List possible predictions.
predictions_ensemble %>% unique

# Test and score.
(rmse_lm1_rt3 <- RMSE(test_set_lm$nextSumAssists, predictions_ensemble))

# Plot the tree.
varImp(fit_lm1_rt3) %>% arrange(desc(Overall))



## ----Linear with Regression Input, warning=FALSE----------------------------------------------------------------------------------------------------


# First, run prediction with RT1 against train set.
pred_rt1 <- predict(fit_rt1, newdata = train_set)

# Add predictions to train set.
train_set_rt1 <- cbind(train_set, pred_rt1)

# Rename added field.
train_set_rt1 <- train_set_rt1 %>% 
  rename(rt1_nextSumAssists="...14")

# See sample rows.
train_set_rt1 %>% head(3)

# Add rt1 predictions to test set.
pred_rt1 <- predict(fit_rt1, newdata=test_set)
test_set_rt1 <- cbind(test_set, pred_rt1)

# Rename added field.
test_set_rt1 <- test_set_rt1 %>% 
  rename(rt1_nextSumAssists="...14")
# See relevant fields in sample rows.
test_set_rt1 %>% 
  select(player_id, sumAssists, nextSumAssists, rt1_nextSumAssists) %>% 
  head(3)

# Build linear model using rt1 prediction as input.
fit_rt1_lm3 <- lm(nextSumAssists~., data = train_set_rt1)

# See new model.
summary(fit_rt1_lm3)

# Make predictions using this new model.
predictions_ensemble2 <- predict(fit_rt1_lm3, newdata = test_set_rt1)

# Test and score.
(rmse_rt1_lm3 <- RMSE(test_set_rt1$nextSumAssists, predictions_ensemble2))



## ----RT Linear impacted players---------------------------------------------------------------------------------------------------------------------

# Table the counts of predictions by RT1 over 36.75.
#  There are 64 estimates this will impact.
table(test_set_rt1$rt1_nextSumAssists) 




## ----Random Forest Standalone Model 1---------------------------------------------------------------------------------------------------------------

# Sample 10 percent of the training set.
#  Create a sample index randomly.
set.seed(1)
sample_index <- createDataPartition(y = train_set$nextSumAssists, times = 1, p = 0.25, list = FALSE)
#  Use the index to sample from training.
sample2 <- train_set[sample_index,]

# Run a random forest model.
#  Runs in roughly 10 minutes on an 8-core CPU with 16G of RAM.
fit_rf1 <- randomForest(nextSumAssists ~ sumAssists + sumPoints + sumShots + 
                        meanAssists + meanShots + 
                        sumTimeOnIce + meanTimeOnIce, data=sample2
                        , na.action=na.fail)

# Show random forest summary.
fit_rf1

# Show important variables.
varImp(fit_rf1) %>% 
  arrange(desc(Overall))



## ----RF1 Results------------------------------------------------------------------------------------------------------------------------------------


# Use model to predict.
predictions <- predict(fit_rf1, newdata=test_set)

# Calculate and save RMSE.
(rmse_rf1 <- RMSE(test_set$nextSumAssists, predictions))




## ----Examine RMSE Score Table-----------------------------------------------------------------------------------------------------------------------

# Put RMSE scores into a vector.
rmse_score <- c(rmse_rt1, rmse_rt2, rmse_lm1, rmse_lm1a
                , rmse_lm1_rt3, rmse_rf1, rmse_rt1_lm3)

# Put RMSE descriptions in a vector.
rmse_description <- c ("Regression Tree 1"
                      ,"Regression Tree 2"
                      ,"Linear Model 1"
                      ,"Linear Model 2"
                      ,"Regression Tree 3, Linear Feature"
                      ,"Random Forest Model"
                      ,"Linear Model, Regression Tree Feature"
                      )
                      
# Display result as a table, sorted by ascending RMSE.
#  Best result on top.
rmse_table <- cbind(rmse_description, round(rmse_score,4)) %>% 
  data.frame() %>% arrange(rmse_score)

# Change column titles.
rmse_table <- rmse_table %>% 
  rename(RMSE= V2, Model = rmse_description) 

# Display results in a formatted table.
rmse_table %>% kable()



## ----Validation Model, warning=FALSE----------------------------------------------------------------------------------------------------------------

# First, run prediction with RT1 against validation set.
pred_v <- predict(fit_rt1, newdata = validation)

# Add predictions to validation set.
validation_set <- cbind(validation, pred_v)

validation_set
# Rename added field to suit the next step of the final model.
#  Since we used this variable name in the original conception
#  of the linear step, we can't vary from it.
validation_set <- validation_set %>% 
  rename(rt1_nextSumAssists="...14")

# See relevant fields in sample rows.
validation_set %>% select(player_id, sumAssists
                , nextSumAssists, rt1_nextSumAssists) %>% 
                  head(3)

# Use the final linear model to make final predictions.
predictions_validation <- predict(fit_rt1_lm3, newdata = validation_set)

# Test and score.
(rmse_validation <- RMSE(validation_set$nextSumAssists, predictions_validation))



## ----Final Code-------------------------------------------------------------------------------------------------------------------------------------
# Put validation predictions and actuals side-by-side in a table.
validatedPredictions <- cbind(validation, predictions_validation) 

# Rename predictions field
validatedPredictions <- validatedPredictions %>% 
  rename(pred_val="...14")

# Create an object with new fields for (rounded)
#  predictions and errors. 
predictionError <- validatedPredictions %>% 
  unique %>%
  mutate(pred=round(pred_val,0)
         ,err=round(nextSumAssists,0)-round(pred_val,0)) %>%
          arrange(abs(err))

# Graph the errors in a scatter plot.
#  Show predictions in color.
#  Draw a green smooth mean line.
#  Adjust scale markers with sequences.
#  Add labels.
predictionError %>% ggplot(aes(x=nextSumAssists, y=err, colour=pred_val)) +
  geom_jitter() + 
  geom_smooth(color="green") +
  scale_y_continuous(breaks = seq(-45,45,5)) +
  scale_x_continuous(breaks = seq(0,100,5)) +
  geom_vline(xintercept = 34, color="red") +
  labs(y="Model Error (Assists)"
      ,x="Actual Player Next Season Assists"
      ,title="Final Model Error by Actual Player Next Season Assists")




## ----Final Model Error Mean and SD------------------------------------------------------------------------------------------------------------------
# Compute mean of errors.
mean(predictionError$err)

# Compute standard deviation of errors.
sd(predictionError$err)



