function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.
    visible_data = sample_bernoulli(visible_data)
    vtoh=visible_state_to_hidden_probabilities(rbm_w,visible_data);
    h1=sample_bernoulli(vtoh);
    htov=hidden_state_to_visible_probabilities(rbm_w,h1);
    v2=sample_bernoulli(htov);
    vtoh=visible_state_to_hidden_probabilities(rbm_w,v2);
    ret=-configuration_goodness_gradient(v2,vtoh)+configuration_goodness_gradient(visible_data,h1);
end
