function [varargout] = qpip(varargin)
%function [vars,status,stats] = qpip(Q,c,C,d,A,b,options)

% QPIP: Implementation of Mehrotra-like Primal-Dual QP solver
% for general quadratic programs.  Based on the C++ solver OOQP.
%
% Usage:
%   [vars,status,stats] = qpip(Q,c,C,d,A,b,options)   
%
%   This finds a solution to the problem:
%
%   min 1/2 x'Qx + c'x
%   s.t. Ax = b, Cx <= d.
%
%   If no equalities (or inequalities) exists, specify and empty
%   matrix for these inputs.   For inequality constrained problems,
%   the usage qpip(Q,c,C,d,options) is also valid.  
%
%   The optional input 'options' is a structure of user definable
%   solver options.  Default values for all fields can be found from
%   options = qpip('defaults');
%   
%   [vars,status,stats] = qpip(data,options) is the same as above,
%   with the problem data contained in a structure 'data' with 
%   fields 'Q','c','C','d','A','b'.  The fields defining the 
%   equalities and/or inequalities are not required.
%
%
% Outputs : 
%   vars   : structure containing the primal solution 'x', dual 
%            variables 'y' and 'z' corresponding to the equality 
%            and inequality multipliers respectively, and slack 
%            variables 's'
%
%   status : 1 if successful, -1 if infeasibility detected, 0 if
%            not converged
%
%   stats  : structure containing possibly useful solver statistics

% Author : P. Goulart, Univ. Oxford.  Last modified 24/07/2017
%
% NB: this code is follows closely the the implemention of the C++
% solver OOQP, but in a pure matlab implementation

%sanity check inputs
narginchk(1,7); nargoutchk(0,3);

%Usage : qpip('defaults').  Returns default settings.
%all other use cases handled by parse_inputs

if(nargin == 1 && ischar(varargin{1}))
    if(strcmp(varargin{1},'defaults'))
        varargout{1} = parse_options(struct);
    else
        error('You probably meant "qpip(''defaults'')"');
    end
    return;
else
    [inputData,options] = parse_inputs(varargin{:});
end

%initialize problem structure and populate 
%with problem data
p = initializeProblem(inputData,options);

%data norm initialize
normD = dataNorm(p.data,inf);

%problem variable initialization
p = defaultStart(p,normD);

%get the complementarity measure mu
muval = calcMu(p);

%suppress factorization warnings.  Do this because
%the LDL factorization code complains about ill
%conditioning near optimality
wstate = warning('off','MATLAB:singularMatrix');

%print a message header
printHeader(p);

%log the start time
tstart = tic;

%-------------------------------------------
%-------------------------------------------
for iter = 1:p.options.max_iter      
    
    %Update the right hand side residuals
    p  = calcResiduals(p);    
    
    %termination test
    status = checkTermination(p,muval,normD);
    if(status), break; end
    
    %-------------------------------------------
    %PREDICTOR STEP   
    %-------------------------------------------

    %find the RHS for this step
    p = calcAffineStepRHS(p,0);     
    %factor and solve
    p = factor(p);
    stepAff = solve(p);    
    %change signs
    stepAff = negateVariables(stepAff);    
    
    %-------------------------------------------
    % Calculate centering parameter
    
    %determine the largest step that preserves
    %consistency of the multiplier constraint  
    alphaAff = stepBound(p,stepAff);
    muAff    = mustep(p,stepAff,alphaAff);
    sigma    = (muAff/muval)^3;
        
    %-------------------------------------------
    %CENTERING-CORRECTOR STEP
    %-------------------------------------------
    
    %Add a corrector (centering) term
    p = calcCorrectorStepRHS(p,stepAff,-sigma*muval);     
    %solve for the corrected system
    stepCC = solve(p);    
    %change signs
    stepCC = negateVariables(stepCC);        
    
    %--------------------------------------------    
    %determine the largest step that preserves
    %consistency of the multiplier constraint    
    alphaMax = stepBound(p,stepCC);    
    
    %a simple blocking mechanism
    stepSize = alphaMax*p.options.gamma;    
    
    %-------------------------------------------
    %take the step and update mu
    p = addToVariables(p,stepCC,stepSize);
    muval = calcMu(p);
    
    %print out some progress information
    printMessages(p,stepSize,iter);
    
end
%-------------------------------------------
%-------------------------------------------
    
%final check for residual convergence
p  = calcResiduals(p);  

%reset matlab warnings
warning(wstate);

%extract the full solution variable set
vars = p.variables;

%collect solve statistics
stats.iterations = iter;
stats.status     = status;
stats.value      = 1/2*(vars.x'*(p.data.Q*vars.x)) + p.data.c'*vars.x;
stats.dualityGap = dualityGap(p);
stats.solveTime  = toc(tstart);
stats.residuals  = p.residuals;

%deal into outputs
varargout = {vars,status,stats};


%------------------------------------------------------------
%-----------------------------------------------------------
function [options] = parse_options(options)

%merge user options with defaults

p = inputParser;
addParameter(p,'max_iter',  100);
addParameter(p,'gamma',     0.99);
addParameter(p,'tolMu',     1e-7);
addParameter(p,'tolR',      1e-7);
addParameter(p,'minPhi',    1e10);
addParameter(p,'verbose',   true);

%update from external options
parse(p,options);
options = p.Results;


%------------------------------------------------------------
%-----------------------------------------------------------
function [inputData,options] = parse_inputs(varargin)

%read user inputs and return in a problem data structure
%and structure of solver options

%check first for the case qpip(problem,options)
if(isstruct(varargin{1}))
    %assume qpip(problem) or qpip(problem,options)
    if(nargin == 1), options = struct; 
    else options = varargin{2}; end
    
    inputData = varargin{1};
    %if no equalities or inequalities are specified,
    %configure with empty matrices
    if(~isfield(inputData,'A'))
        inputData.A = []; 
        inputData.b = [];
    end
    if(~isfield(inputData,'C'))
        inputData.C = []; 
        inputData.d = [];
    end

%next check the cases:
%    qpip(Q,c,C,d,options)
%and qpip(Q,c,C,d,A,b,options)

elseif(nargin <= 5) 
    %assume qpip(Q,c,C,d) or qpip(Q,c,C,d,options)
    if(nargin == 4), options = struct; 
    else options = varargin{5}; end
    
    [inputData.Q,inputData.c,...
     inputData.C,inputData.d,...
     inputData.A,inputData.b] = deal(varargin{1:4},[],[]);
    
else
    %assume qpip(Q,c,C,d,A,b) or qpip(Q,c,C,d,A,b,options)
    if(nargin == 6), options = struct; 
    else options = varargin{7}; end
    
    [inputData.Q,inputData.c,...
     inputData.C,inputData.d,...
     inputData.A,inputData.b] = deal(varargin{1:6});
end

%try to work out the number of variables
%from the 'q' field, which must exist.  Then
%resize any empty inputs so that they are
%compatible in the column dimension

nvar = length(inputData.c);
if(isempty(inputData.A))
    inputData.A = sparse(0,nvar);
    inputData.b = zeros(0,1);
end
if(isempty(inputData.C))
    inputData.C = sparse(0,nvar);
    inputData.d = zeros(0,1);
end
if(isempty(inputData.Q))
    inputData.Q = sparse(nvar,nvar);
end

%shape all vectors into column form
inputData.c = inputData.c(:);
inputData.b = inputData.b(:);
inputData.d = inputData.d(:);

%merge user options with defaults
options = parse_options(options);

%check the problem dimensions
assert(size(inputData.Q,1) == size(inputData.Q,2),'Input ''Q'' is not square');
assert(size(inputData.Q,1) == length(inputData.c),'Inputs ''Q'' and ''c'' have incompatible dimension');
assert(size(inputData.A,2) == length(inputData.c),'Input ''A'' has incompatible dimension');
assert(size(inputData.C,2) == length(inputData.c),'Input ''C'' has incompatible dimension');
assert(size(inputData.A,1) == length(inputData.b),'Inputs ''A'' and ''b'' have incompatible dimension');
assert(size(inputData.C,1) == length(inputData.d),'Inputs ''C'' and ''d'' have incompatible dimension');

    
%------------------------------------------------------------
%------------------------------------------------------------
function printHeader(p)

%Print some problem data and other header information 

if(~p.options.verbose), return; end

fprintf('------------------------------------------------------------\n');
fprintf('::: QP interior point solver :::\n');
fprintf('Author                : P. Goulart, Univ. Oxford\n');
fprintf('Number of variables   : %i \n',p.data.nx);    
fprintf('Number of equalities  : %i \n',p.data.ny);    
fprintf('Number of inequalities: %i \n',p.data.nz);    
fprintf('------------------------------------------------------------\n');

%------------------------------------------------------------
%------------------------------------------------------------
function printMessages(p,stepSize,iter)

%Print some progress messages

if(~p.options.verbose), return; end

fprintf('Iteration %2i: ',iter);
fprintf('Step Size: %.4e, ',stepSize);
fprintf('Duality Gap: %.4e, ',dualityGap(p));
fprintf('Residual Norm : %.4e',residualNorm(calcResiduals(p),inf));
fprintf('\n');


%------------------------------------------------------------
%------------------------------------------------------------
function status = checkTermination(p,muval,normD)

%test for convergence or infeasibility

gap      = abs(dualityGap(p));
normR    = residualNorm(p,inf);
phi      = (normR + gap)./normD;
minPhi   = min(p.options.minPhi,phi);

if(muval <= p.options.tolMu && normR <= p.options.tolR*normD) 
    %convergence.  Exit.
    if(p.options.verbose)
        fprintf('Optimization Successful\n');
    end
    status = 1;
    
elseif(phi > 10-8 && phi > 10^4*minPhi)
    %infeasible; Exit;
    if(p.options.verbose)
        fprintf('Optimization Failed\n');
    end
    status = -1;
else
    status = 0;
end

%------------------------------------------------------------
%------------------------------------------------------------
function mu = calcMu(p)

%calculate complementarity measure 

z = p.variables.z;
s = p.variables.s;
m = length(z);

if(m == 0) %no inequalities case
    mu = 0;
else
    mu = sum(abs(z.*s))./m;
end

%------------------------------------------------------------
%------------------------------------------------------------
function gap = dualityGap(p)

%dualitygap: Calculate duality gap

%the duality gap is defined as
%gap = x'Qx + c'x + b'y + d'z

%problem data
[Q,b,c,d] = deal(p.data.Q, p.data.b, p.data.c, p.data.d);

%variables
[x,y,z] = deal(p.variables.x, p.variables.y, p.variables.z);   

%calculate the gap
gap = x'*(Q*x) + c'*x + b'*y + d'*z;


%------------------------------------------------------------
%------------------------------------------------------------
function [mu,m] = mustep(p,step,alpha)

%mustep: calculate the value of z's/m given 
%a step in this input direction

%the current system z and s
z = p.variables.z;
s = p.variables.s;

%the proposed z and s directions
dz = step.z;
ds = step.s;

%number of variables
m = length(s)';

mu = sum(abs(z+alpha.*dz).*(s+alpha.*ds))./m;    

%------------------------------------------------------------
%------------------------------------------------------------
function p = factor(p)

%update the jacobian
p = updateJacobian(p);

%LDL' factorization
[L,D,perm,S] = ldl(p.linsys.J,'vector');

p.linsys.factors.L = L;
p.linsys.factors.D = D;
p.linsys.factors.perm  = perm;
p.linsys.factors.scale = diag(S);


%------------------------------------------------------------
%------------------------------------------------------------
function variables = solve(p)
%solve: Solve the Newton system for a given
%set of residuals, using the current Jacobian
%factorization

%get the Jacobian factorization
L = p.linsys.factors.L;
D = p.linsys.factors.D;
perm  = p.linsys.factors.perm;
scale = p.linsys.factors.scale;

%construct the rhs to be solved (including any pre-elimination)
rhs = stackResiduals(p);

%solve it
lhs(perm) = L'\(D\(L\(rhs(perm).*scale(perm)))).*scale(perm);
lhs = lhs(:); %force a column

%parse the solution (including any post-solving)
%back into a *copy* of the variables
variables = splitVariables(p,lhs);

%Some extra information
%disp(sprintf('Solve. norm(rhs) = %g, norm(lhs) = %g',norm(rhs),norm(lhs)));


%------------------------------------------------------------
%------------------------------------------------------------
function rhs = stackResiduals(p)

%stackResiduals: Stack all of the residuals into
%a vector for use as a RHS in a linear solve

%get the current residuals
rQ = p.residuals.rQ;
rC = p.residuals.rC;
rA = p.residuals.rA;
rS = p.residuals.rS;

%get the Z terms from the variables
z  = p.variables.z;

%eliminate the rS terms
rC = rC - rS./z;

%put them all together
rhs = [rQ;rA;rC];



%------------------------------------------------------------
%------------------------------------------------------------
function outVariables = splitVariables(p,lhs)

%splitVariables: Split a solution vector into
%its components variables based on their ordering.

%the output should have the same structure as
%the input (maintains indexing etc)
outVariables = p.variables;

%Some shorter names for convenience
[nx,ny,nz] = deal(p.data.nx, p.data.ny, p.data.nz);  

%parse it up
dx = lhs(1:nx);
dy = lhs((1:ny) + nx);
dz = lhs((1:nz) + nx + ny);
outVariables.x(:) =  dx;
outVariables.y(:) =  dy;                                
outVariables.z(:) =  dz;  

%post-solve solve for the ds terms
%using the Z and S terms from the 
%variables, plus the current rS
%in the residuals
rS = p.residuals.rS;
z  = p.variables.z;
s  = p.variables.s;

outVariables.s(:) = (rS-s.*dz)./z;  



%------------------------------------------------------------
%------------------------------------------------------------
function p = updateJacobian(p)

%pdate the diagonal parts of the Jacobian matrix

z = p.variables.z;
s = p.variables.s;
sigma = (s./z);

p.linsys.J(p.linsys.idxSigma) = -sigma;


%------------------------------------------------------------
%------------------------------------------------------------
function val = dataNorm(data,varargin)

%a few shortcuts
[Q,A,C,b,c,d] = deal(data.Q, data.A, data.C, ...
                     data.b, data.c, data.d);  
                 
%vectorize the data and use the non-infinite values only                 
vecData = [A(:);b(:);C(:);d(:);Q(:);c(:)];
if(any(isinf(vecData)))
    error('Infinite valued problem data not supported');
end

%construct the norm
val = norm(vecData,varargin{:});




%------------------------------------------------------------
%------------------------------------------------------------
function val = residualNorm(p,varargin)

%residualNorm: Returns residual vector norms 
%Examples: residualNorm(obj,inf) = max(abs(residuals))
%          residualNorm(obj,1) = sum(abs(residuals))
%
%See also: norm

%a few shortcuts
rQ  = p.residuals.rQ;
rA  = p.residuals.rA; 
rC  = p.residuals.rC;

vec = [rQ(:);rA(:);rC(:)];
val = norm(vec,varargin{:});


%------------------------------------------------------------
%------------------------------------------------------------
function p = initializeProblem(problemData,options)

%create all data structures

%cook up an initial solution
p.data      = initializeData(problemData);
p.variables = initializeVariables(p);
p.residuals = initializeResiduals(p);
p.linsys    = initializeLinSys(p);
p.options   = options;

%------------------------------------------------------------
%------------------------------------------------------------
function var = initializeVariables(p)

%Some shorter names for convenience
[nx,ny,nz,ns] = deal(p.data.nx, p.data.ny, p.data.nz, p.data.ns);  

%create vector variables of the right size
var.x = zeros(nx,1);  
var.y = zeros(ny,1);  
var.z = ones(nz,1);  
var.s = ones(ns,1);  

%------------------------------------------------------------
%------------------------------------------------------------
function linsys = initializeLinSys(p)

%a few shortcuts
[Q,A,C] = deal(p.data.Q, p.data.A, p.data.C); 
                 
[nx,ny,nz,ns] = deal(p.data.nx, p.data.ny, ...
                     p.data.nz, p.data.ns);  

%Create the Jacoban matrix with a dummy Sigma
S  = -speye(ns);
Z1 = spalloc(ny,ny,0);
Z2 = spalloc(nz,ny,0);
Z3 = Z2';
ZA = spalloc(nx,ny,0);
ZC = spalloc(nx,ns,0);

%construct the jacobian.  Since ldl is used for 
%factorization, only the lower part is needed
linsys.J = ...
    [Q  ZA  ZC;
     A  Z1  Z3;
     C  Z2   S];
        
%get the indices for the entries of S
idx = (nx + ny) + (1:ns);
linsys.idxSigma = sub2ind(size(linsys.J),idx,idx);

%configure structure for factors
linsys.factors.L = [];
linsys.factors.D = [];
linsys.factors.perm  = [];
linsys.factors.scale = [];

%------------------------------------------------------------
%------------------------------------------------------------
function res = initializeResiduals(p)

%initializeResiduals: Initializes the residuals
%vector, and creates various internal data
%that helps in manipulating it 

%Some shorter names for convenience
nx = p.data.nx;  
ny = p.data.ny;  
nz = p.data.nz;  
ns = p.data.ns;  

%create matrices of the right size
%to hold the residual components
res.rQ  = zeros(nx,1);
res.rA  = zeros(ny,1);
res.rC  = zeros(nz,1);
res.rS  = zeros(ns,1);


%------------------------------------------------------------
%------------------------------------------------------------
function data = initializeData(problemData)

%package up all the data
data = problemData;

%sparsify everything
data.Q = sparse(data.Q);
data.A = sparse(data.A);
data.C = sparse(data.C);

%a few sizing parameters
data.nx = size(data.A,2);
data.ny = size(data.A,1);
data.nz = size(data.C,1);
data.ns = data.nz;

%------------------------------------------------------------
%------------------------------------------------------------
function p = calcResiduals(p)

%calcResiduals: Calculates the problem
%residuals based on the current variables

%Recall that for the nominal problem
%min x'Qx + c'x
%Ax = b ,Cx <= d
%
%the residuals are defined as
% rQ  = (Q*x + c + A'*y + C'*z);
% rA  = A*x - b;
% rC  = C*x - d + s;

%problem data
[Q,A,C,b,c,d] = deal(p.data.Q, p.data.A, p.data.C, ...
                     p.data.b, p.data.c, p.data.d);

%variables
[x,y,z,s] = deal(p.variables.x, p.variables.y, ...
                 p.variables.z, p.variables.s);

%calculate the residuals
rQ  = Q*x + A'*y + C'*z + c;
rA  = A*x - b;
rC  = C*x + s - d;

%update the residuals
p.residuals.rQ    = rQ;
p.residuals.rC    = rC;
p.residuals.rA    = rA;
p.residuals.rS(:) = 0;


%------------------------------------------------------------
%------------------------------------------------------------
function var = negateVariables(var)

%change signs in a variables structures
var.x = -var.x;
var.y = -var.y;
var.z = -var.z;
var.s = -var.s;


%------------------------------------------------------------
%------------------------------------------------------------
function p = addToVariables(p,var,alpha)

%shift the problem variables by a scaled correction term
p.variables.x = p.variables.x + alpha.*var.x;
p.variables.y = p.variables.y + alpha.*var.y;
p.variables.z = p.variables.z + alpha.*var.z;
p.variables.s = p.variables.s + alpha.*var.s;


%------------------------------------------------------------
%------------------------------------------------------------
function p = calcAffineStepRHS(p,shift)

%Adjust the RHS terms for a pure newton step.
%Assumes residuals (rQ,rA,rC) have already
%been calculated

z = p.variables.z;
s = p.variables.s;

%update the final term
p.residuals.rS(:) = z.*s + shift;


%------------------------------------------------------------
%------------------------------------------------------------
function p = calcCorrectorStepRHS(p,stepAff,shift)

%ad_rS_ZS_alpha: adds to the rS component of the
%residuals a term =  dZ*dS*e + shift*e

dz = stepAff.z;
ds = stepAff.s;

%update the residuals
p.residuals.rS(:) = p.residuals.rS(:) + dz.*ds + shift;


%------------------------------------------------------------
%------------------------------------------------------------
function p = defaultStart(p,normD)
%default first step

%find some interior point (large z and s)
sdatanorm = sqrt(normD);
p.variables.z(:) = sdatanorm;
p.variables.s(:) = sdatanorm;


p = calcResiduals(p);   
p = calcAffineStepRHS(p,0);  

%factor and solve
p    = factor(p);
step = solve(p);    
step = negateVariables(step); 

%take the full affine scaling step
p = addToVariables(p,step,1.0);

%shift the bound variables
shift = 1e3 + violation(p);
p = shiftBoundVariables(p,shift,shift);



%------------------------------------------------------------
%------------------------------------------------------------
function v = violation(p)

%Find the maximum constraint violation

C = p.data.C;
d = p.data.d;
x = p.variables.x;
z = p.variables.z;
s = p.variables.s;

v = max(0,max(-[z;s]));


%------------------------------------------------------------
%------------------------------------------------------------
function p = shiftBoundVariables(p,alpha,beta)

%shift the bound variables

p.variables.z = p.variables.z + alpha;
p.variables.s = p.variables.s + beta;


%------------------------------------------------------------
%------------------------------------------------------------
function [alpha] = stepBound(p,step)

%stepbound: calculate the maximum allowable
%step in the proposed direction (in [0 1])

%the current system z and s
z = p.variables.z;
s = p.variables.s;

%the proposed z and s directions
dz = step.z;
ds = step.s;

if(any([dz;ds]<0))
    tmp   = [dz;ds]./[z;s];
    alpha = min(max(0,1/max(-tmp)),1);
else
    alpha = 1;
end

