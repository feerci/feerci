FEERCI: A Package for Fast non-parametric confidence intervals for Equal Error Rates
******************************************


**FEERCI** is an opinionated, easy-to-use package for calculating EERs and non-parametric confidence intervals efficiently. It offers a single method, ``feerci.feerci``, that returns both an EER and CI for provided impostor and genuine scores. The only dependency is numpy.

Installation
=======
``pip install feerci``

What's New
=======
0.1.0
--------
- Initial release of package


License
=====
**FEERCI** is distributed under the MIT license

Version
=====
0.1.0

Examples
======
Calculating just an EER::

    import feerci
    import numpy as np
    impostors = np.random.rand(100)
    genuines = np.random.rand(100)
    eer,_,_,_ = feerci.feerci(impostors,genuines,is_sorted=False,m=-1)

Calculating an EER and 95% confidence interval::

    eer,bootstrapped_eers,ci_lower,ci_upper = feerci.feerci(impostors,genuines,is_sorted=False)

Calculating an EER and 99% confidence interval::

    eer,bootstrapped_eers,ci_lower,ci_upper = feerci.feerci(impostors,genuines,is_sorted=False,ci=.99)

Calculating an EER and 99% confidence interval on 1000 bootstrap iterations::

    eer,bootstrapped_eers,ci_lower,ci_upper = feerci.feerci(impostors,genuines,is_sorted=False,m=1000,ci=.99)

