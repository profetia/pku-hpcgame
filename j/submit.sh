#!/usr/bin/bash

rm -rf hpl-mxp submission.tar
mkdir -p hpl-mxp
cp *.c *.h *.sh Makefile hpl-mxp
tar -cvf submission.tar hpl-mxp
