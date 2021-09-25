
cd "/Users/yoshi/Dropbox/Google Drive/Coding/Python/Linear_Regression/Multiple8"



####Multiple Linear Regression & Single Linear Regression

#Raw Data X vs Raw Data Y
#python3 mlrrwstnmslr.py df.csv Y RM r

#Standardized Data X vs Raw Data Y
python3 mlrrwstnmslr.py df.csv Y RM s

#Normalized Data X vs Raw Data Y
#python3 mlrrwstnmslr.py df.csv Y RM n



####Multiple Linear Regression with specified Xs (e.g., RM, LSAT)

python3 othermlr.py Xconv_Y_Ypred_YpredSingle.csv clifIntercept.txt clifCoefficients.txt Y RM LSTAT