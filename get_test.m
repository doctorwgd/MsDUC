function [ net ] = get_test_ms(varargin)
addpath(genpath('src\'));
% cpu and gpu settings 
opts.idx_gpus = 1; % 0: cpu
opts.num_images = 32; % images to be generated
opts.matconvnet_path = 'F:\matlab\matconvnet-1.0-beta25\matlab\vl_setupnn.m';
opts.net_path = 'F:\matlab\matconvnet-1.0-beta25\initial_net8_20\initial_net8_20.mat';

opts = vl_argparse(opts, varargin);
run(opts.matconvnet_path);

%% load network
net = load(opts.net_path);
net = net.net(1); % idx 1: Generator, 2: Discriminator
net = dagnn.DagNN.loadobj(net);
net.mode = 'test';

if opts.idx_gpus >0
    gpuDevice()
    net.move('gpu');
end

rng('default')
load('Testdata\test\sample.mat');
data = single(x);
randn('seed',0);

stripe_mat = single(repmat(20.*rand(1,size(data,2)*10),size(data,1),1)/255);
stripe_cub = reshape(stripe_mat, size(x));
noise = data + stripe_cub ;

PSNR_noise  =  csnr( 256*(data), double(256*(noise)), 0, 0 )
SSIM_noise  =  cal_ssim( 256*(data), double(256*(noise)), 0, 0 )

if opts.idx_gpus >0,   noise = gpuArray(noise);    end

tic
net.eval({'input',noise});
im_out = gather(net.vars(net.getVarIndex('conv8')).value); 
toc

output = noise - im_out;
PSNR_CNN  =  csnr( 256*(data), double(256*(output)), 0, 0 )
SSIM_CNN  =  cal_ssim( 256*(data), double(256*(output)), 0, 0 )
figure,imshow([noise(:,:,1),output(:,:,1)])

fprintf('get_test_HSI restoration is complete\n');

return;