library(arrow)
library(survival)

# Set up folders
outfolder <- paste("/proj/characklab/flooddata/NC/multiple_events/analysis/",Sys.Date(),"_loan_survival_analysis",sep="")
dir.create(outfolder)

# Read in Fannie Mae / Freddie Mac single-family loan performance data
performance_path = "/proj/characklab/flooddata/NC/multiple_events/financial_data/GSEs/fannie_freddie_SF_loan_level_performance.parquet"

usecols <- c("loan_id","loan_purpose","occupancy_status","loan_term","loan_age","prepayment","default","spread_vs_MORTGAGE30US","spread_vs_MORTGAGE15US")
df <- read_parquet(performance_path,col_select=usecols)

## (!) comment out once debugged
#loan_ids <- unique(df$loan_id)
#selected_loans <- sample(loan_ids,100000)
#df <- df[df$loan_id %in% selected_loans,]

# Drop loans from non-primary residences
df <- df[df$occupancy_status=="Primary",]

# Drop those that default since we're more interested in prepayment
# (this is a very small proportion of all loans)
df <- df[!(df$loan_id %in% df$loan_id[df$default==1]),]
df <- df[(df$loan_age > 0),]

df$time1 <- df$loan_age - 1
df$time2 <- df$loan_age

# Subset by loan type
purchase30 <- df[(df$loan_purpose=="Purchase")&(df$loan_term==360),]
refinance30 <- df[(df$loan_purpose=="Refinance")&(df$loan_term==360),]
refinance15 <- df[(df$loan_purpose=="Refinance")&(df$loan_term==180),]

# Specify relevant rate spread to use as predictor of prepayment
purchase30$rate_spread <- purchase30$spread_vs_MORTGAGE30US
refinance30$rate_spread <- refinance30$spread_vs_MORTGAGE30US
refinance15$rate_spread <- refinance15$spread_vs_MORTGAGE15US

print(paste("Number of purchase30 loan-month observations:",nrow(purchase30)))
print(paste("Number of refinance30 loan-month observations:",nrow(refinance30)))
print(paste("Number of refinance15 loan-month observations:",nrow(refinance15)))

print(paste("Number of purchase30 prepayment events:",sum(purchase30$prepayment,na.rm=TRUE)))
print(paste("Number of refinance30 prepayment events:",sum(refinance30$prepayment,na.rm=TRUE)))
print(paste("Number of refinance15 prepayment events:",sum(refinance15$prepayment,na.rm=TRUE)))

# Fit cox proportional hazard models to data
p30_coxph <- coxph(Surv(time1, time2, prepayment) ~ rate_spread, id=loan_id, data=purchase30)
r30_coxph <- coxph(Surv(time1, time2, prepayment) ~ rate_spread, id=loan_id, data=refinance30)
r15_coxph <- coxph(Surv(time1, time2, prepayment) ~ rate_spread, id=loan_id, data=refinance15)

# Extract information needed to quantify baseline hazard function (i.e., hazard rate when rate spread is zero)
p30_surv <- survfit(p30_coxph,newdata=data.frame(rate_spread=0))
r30_surv <- survfit(r30_coxph,newdata=data.frame(rate_spread=0))
r15_surv <- survfit(r15_coxph,newdata=data.frame(rate_spread=0))

# Convert to dataframe
p30_surv_df <- with(p30_surv, data.frame(time, n.risk, n.event, surv, cumhaz, lower, upper))
r30_surv_df <- with(r30_surv, data.frame(time, n.risk, n.event, surv, cumhaz, lower, upper))
r15_surv_df <- with(r15_surv, data.frame(time, n.risk, n.event, surv, cumhaz, lower, upper))

# Add info on coefficient of rate spread
p30_surv_df$rate_spread_coeff <- p30_coxph$coefficients['rate_spread']
p30_surv_df$rate_spread_coeff_lower <- confint(p30_coxph,level=0.95)[1]
p30_surv_df$rate_spread_coeff_upper <- confint(p30_coxph,level=0.95)[2]

r30_surv_df$rate_spread_coeff <- r30_coxph$coefficients['rate_spread']
r30_surv_df$rate_spread_coeff_lower <- confint(r30_coxph,level=0.95)[1]
r30_surv_df$rate_spread_coeff_upper <- confint(r30_coxph,level=0.95)[2]

r15_surv_df$rate_spread_coeff <- r15_coxph$coefficients['rate_spread']
r15_surv_df$rate_spread_coeff_lower <- confint(r15_coxph,level=0.95)[1]
r15_surv_df$rate_spread_coeff_upper <- confint(r15_coxph,level=0.95)[2]

# Save results
saveRDS(p30_coxph,file=paste(outfolder,"purchase30_coxph.rds",sep="/"))
saveRDS(r30_coxph,file=paste(outfolder,"refinance30_coxph.rds",sep="/"))
saveRDS(r15_coxph,file=paste(outfolder,"refinance15_coxph.rds",sep="/"))

write.csv(p30_surv_df,paste(outfolder,"purchase30_survival_params.csv",sep="/"),row.names=FALSE)
write.csv(r30_surv_df,paste(outfolder,"refinance30_survival_params.csv",sep="/"),row.names=FALSE)
write.csv(r15_surv_df,paste(outfolder,"refinance15_survival_params.csv",sep="/"),row.names=FALSE)




