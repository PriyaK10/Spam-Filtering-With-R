dataset=read.csv(file="C:/Users/1542283/Desktop/R/Mall_Customers.csv", 
                 stringsAsFactors=FALSE)
anyNA(dataset)
unique(dataset$Genre)
str(dataset)
dataset$Genre=factor(dataset$Genre,levels = c("Male","Female"),labels = c(0,1))
str(dataset)
names(dataset)
colnames(dataset)[4]="AnnualIncome"
colnames(dataset)[5]="SpendingScore"
colnames(dataset)       
range(dataset$Age)
maindf=dataset[,2:5]
str(maindf)
set.seed(6)
is.double(maindf$Age)
# using optimal elbow rule to find optimal number of clusters
maindf$Age=cut(maindf$Age, breaks= 4, labels=FALSE)
maindf$Age=as.factor(maindf$Age)
anyNA(maindf)
wcss=c()
for(i in 1:10){
  wcss[i]=sum(kmeans(maindf1,i)$withinss)
}
plot(1:10, wcss, type="b", xlab= 'number of clusters')

model=kmeans(maindf1,3,iter.max=10,nstart=10)
model$centers
view(maindf)
maindf1=maindf[,c(2,4)]
library(cluster)
clusplot(maindf1,model$cluster, lines=, shade=T, color=T,labels=2,span=T,plotchar=F)
help(spread)

x=matrix(1:6,2,3)
x
x[]=0
x


# kmeans in IRIS
data1=iris
library(ggplot2)
ggplot(data1, aes(data1$Petal.Length,data1$Petal.Width,color=Species))+geom_point()
# using elbow method to find optimal number of cluster
set.seed(20)
df=data1[,3:4]
wcss=c()
for(i in 1:5){
  wcss[i]=sum(kmeans(df,i)$withinss)
}
plot(1:5, wcss, type="b", xlab= 'number of clusters')
iriscluster=kmeans(df, 3, iter.max=10, nstart=20)
iriscluster$cluster
iriscluster$centers
iriscluster$withinss
wcss
table(iriscluster$cluster,iris$Species)
str(iris)
ggplot(iris, aes(Petal.Length, Petal.Width, color = iriscluster$cluster)) + geom_point()
