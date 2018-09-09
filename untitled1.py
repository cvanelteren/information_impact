#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 13:20:40 2018

@author: casper
"""

def chkTravellerAge():
    while True:
        travAge = input('How old are you? ')
        try:
            travAge = int(travAge)
            if 0 < travAge < 100:
                if travAge < 16:
                    print("You'll receive a 50 percent discount!")
                break
            else:
                print('Please enter a valid age')  
        except:
            print('Please enter an integer')       
    return travAge
chkTravellerAge()