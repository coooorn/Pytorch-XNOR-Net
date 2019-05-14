#! /bin/bash
make

if [ -s ../../binop ]
then
    mv binop/__init__.py ../../binop
    mv binop/_binop.so ../../binop
else
    mv binop ../../binop
fi
rm -rf binop
make clean