#!/bin/sh
numOfArgs=$#
if [ $numOfArgs -ne 1 ]; then
  echo -e "Usage: \nbash $0 dirForCount"
  exit -1
fi
# args
ROOTDIR=$1
# core part
find $ROOTDIR -maxdepth 1 -type d | sort | while read dir; do
count=$(find "$dir" -type f | wc -l)
echo "$dir: $count"
done