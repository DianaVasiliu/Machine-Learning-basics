import numpy as np
import matplotlib.pyplot as plt
plt.style.use('default')
plt.rcParams['figure.dpi'] = 70


# ## Basic plotting - sine & cosine functions

# x = values between 0 and 3*pi, with a step of 0.1
x = np.arange(0, 3 * np.pi, 0.1)
# points on the sine curve
y = np.sin(x)

plt.plot(x, y)

# axis labels
plt.xlabel('x axis label')
plt.ylabel('y axis label')

# graph title
plt.title('Sine')

# legend
plt.legend(['Sine'])

# showing the plot
plt.show()


# ### Plotting with different type of line

plt.plot(x, y, 'o')
plt.show()

plt.plot(x, y, '--')
plt.show()


# ### Plotting multiple functions on the same graphic

x = np.arange(0, 3 * np.pi, 0.1)
y_1 = np.sin(x)
y_2 = np.cos(x)

# plotting 2 functions
plt.plot(x, y_1)
plt.plot(x, y_2)

plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])
plt.show()


# ### Plotting multiple functions in different figures

x = np.arange(0, 3 * np.pi, 0.1)
y_1 = np.sin(x)
y_2 = np.cos(x)

# defining first figure
first_plot = plt.figure(1)
plt.plot(x, y_1)
plt.title('Sine')
plt.legend(['Sine'])

# defining second figure
second_plot = plt.figure(2)
plt.plot(x, y_2)
plt.title('Cosine')
plt.legend(['Cosine'])

plt.show()


# ### Plotting multiple graphics in the same figure, but separately

x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# creating a grid with height=2 and width=1
# and setting the first subplot as active
plt.subplot(2, 1, 1)

# plotting first values
plt.plot(x, y_sin)
plt.title('Sine')

# setting the second subplot as active
plt.subplot(2, 1, 2)

# plotting second values
plt.plot(x, y_cos)
plt.title('Cosine')

plt.show()
