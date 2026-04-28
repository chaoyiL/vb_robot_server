import numpy as np
a = "0.14175761011829957, 0.9601673787806352, -0.23749331035624502, -0.03975314119795767"

a = np.array(a.split(", ")).astype(float)
print(a)

# check it is [x, y, z, w] or [x, y, w, z]