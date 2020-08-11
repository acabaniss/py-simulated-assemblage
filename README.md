# py-simulated-assemblage
Generate simulated assemblages with the same artifact density and frequency as observed assemblages to estimate typicality.


## About
This code takes artifact distributions broken down by categories (types, ware, size, traces, preservation, or other discrete categories) and generates statistically similar assemblages. These assemblages then enable the typicality of a distribution to be assessed, i.e. how different is the observed assemblage from statistically similar assemblages?

This is very similar to a chi-square test, except conducted empirically and without any normality or non-zero assumptions. 

## Use case
These functions will likely be useful if you wish to assess whether particular assemblages are what would be "expected" compared to all other assemblages being considered based on the frequency of different types of stone tools found. For example, does the frequency of different types of tools in House A suggest it is "atypical" compared to other domestic stone tool assemblages?

It can also be used to compare the typicality of different assemblages. For example, which domestic assemblages appear to be the most or least typical in the frequency of goat anatomical parts with butcher marks?

Finally, sets of assemblages can be assessed for typicality. For example, does the set of all domestic assemblages at Site 1 have ceramic vessel frequencies that are close to the average, or are there substantial differences across the site?


