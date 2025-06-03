mkdir datasets
FILE=$1

if [[ $FILE != "apple2orange" &&  $FILE != "horse2zebra"]]; then
    echo "Available datasets are: apple2orange, horse2zebra"
    exit 1
fi

URL=https://github.com/akanametov/cyclegan/releases/download/1.0/$FILE.zip
ZIP_FILE=./datasets/$FILE.zip
TARGET_DIR=./datasets/$FILE/
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_DIR
unzip $ZIP_FILE -d ./datasets/
rm $ZIP_FILE
