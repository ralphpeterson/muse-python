
import time
# %% 
tic = time.time()

n = 20 # number of points in x,y

xs = np.linspace(0+x_dim/n, x_dim*(1-1/n), n)
ys = np.linspace(0+y_dim/n, y_dim*(1-1/n), n)

# Loop over different sound source position
xy_err = np.zeros([n,n])
for i in range(n):
    for j in range(n):
        # Add a speaker (from the same height in z)
        speaker_pos = (xs[i], ys[j], z_pos)
        # Play audio and simulate
        room = fun.generate_room(x_dim, y_dim, z_dim, materials)
        fun.place_mics(room, mic_pos)
        sr_stimulus, stimulus = fun.add_source(speaker_pos, stimulus, sr_stimulus, room)
        room.simulate()
        
        audio,fs,f_lo,f_hi,temp,x_grid,y_grid,in_cage,R = fun.set_MUSE_params(room, x_dim, y_dim, mic_pos, sr, 0.0025)

        del room

        # Run MUSE
        print('Running MUSE:'+str(i*n+j)+'/'+str(n*n))
        r_est, _, rsrp_grid, _, _, _, _, _, _ \
        = muse.r_est_from_clip_simplified(audio,
            fs,
            f_lo, f_hi,
            temp,
            x_grid, y_grid,
            in_cage,
            R,  # Expected to have shape (3, n_mics)
            1)
        xy_err[i][j] = np.sqrt(sum((r_est.T[0] - speaker_pos[:2])**2))
        print(xy_err[i][j])

toc = time.time()
print('Elapsed time: ' + str(toc-tic))
# %% Make a heatmap & histogram
x,y = np.meshgrid(xs, ys)

fig, ax = plt.subplots(figsize=(7,7))
c = ax.pcolormesh(x,y,xy_err*100, cmap='viridis')
ax.set_xlim(0, x_dim), ax.set_ylim(0, y_dim)
cbar = plt.colorbar(c)
cbar.ax.set_ylabel('error (cm)', rotation=0)

ax.set_xlabel('x (m)'), ax.set_ylabel('y (m)')
ax.set_title('$\Delta d_{max} = %1.3f$ cm'%np.amax(xy_err*100)
             +', $z_{mic}$=%1.2f m'%mic_pos.T[0,2])

# Add microphone positions
norm=np.sqrt(sum((mic_dir[0,:2]-mic_pos.T[0,:2])**2))*2

for i in range(n_mic):
    ax.plot(mic_pos[0,i], mic_pos[1,i], 'o', color='r')
    ax.arrow(mic_pos.T[i,0],mic_pos.T[i,1], 
             mic_dir[i,0]/norm, mic_dir[i,1]/norm, 
             head_width=0.05, head_length=0.05, fc='k', ec='k')

ax.set_aspect('equal', 'box')
plt.show()

print('d_max = %1.3f cm'%np.amax(xy_err*100))
print('d_min = %1.3f cm'%np.amin(xy_err*100))
print('d_mean = %1.3f cm'%np.mean(xy_err[:]*100))
print('d_std = %1.3f cm'%np.std(xy_err[:]*100))

# %% Make a histogram of the errors
plt.figure()
plt.hist(xy_err.flatten()*100)
plt.xlabel('error (cm)')
plt.ylabel('counts')
plt.show()




