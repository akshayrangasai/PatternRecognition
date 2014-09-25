function z = multigauss (x1, x2, k, mu, sigma)
    z = (1/((2*pi)^(-k/2) * det(sigma)^0.5)) * exp(-0.5*bsxfun(@minus,[x1 x2],mu)*inv(sigma)*bsxfun(@minus,[x1 x2],mu)');
end