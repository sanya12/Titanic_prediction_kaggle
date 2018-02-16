import numpy as np
from sklearn import tree
import pandas
from sklearn.ensemble import RandomForestClassifier
#titanic=pandas.read_csv('C:\\Users\FAIYAZ\Downloads\\train.csv')
titanic=pandas.read_csv('data/train.csv')


test=pandas.read_csv("data/test.csv")
#print(titanic["Age"].unique())
#filling up the missing age dataset
#print(titanic["Age"].median )
titanic["Age"]=titanic["Age"].fillna(titanic["Age"].median())
#print(titanic["Age"].median)
#print(titanic.head(5))

#converting to numeric data
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"]=="female","Sex"]=1

#embarked to numeric data
titanic["Embarked"]=titanic["Embarked"].fillna("S")
titanic.loc[titanic["Embarked"]=="S","Embarked"]=0
titanic.loc[titanic["Embarked"]=="C"  ,"Embarked"]=1
titanic.loc[titanic["Embarked"]=="Q","Embarked"]=2

#print(titanic)

"""
#check male female survival rate
print(titanic["Survived"].value_counts())
#no of males who survived and dint survive
print(titanic["Survived"][titanic["Sex"]==0].value_counts())
#no of males
print(titanic["Sex"].value_counts())

"""

titanic["child"]="Nan"
titanic["child"][titanic["Age"]<=18]=1
titanic["child"][titanic["Age"]>18]=0
print(titanic["child"].value_counts())
print(titanic["Survived"][titanic["child"]==1].value_counts())
#print(titanic)

#using decission trees and fitting
target =titanic["Survived"].values
features_one = titanic[["Pclass", "Sex", "Age", "Fare"]].values
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one,target)
print(my_tree_one.feature_importances_)
print(my_tree_one.score(features_one,target))