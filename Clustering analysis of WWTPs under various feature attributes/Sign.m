function [sign]=Sign(Antenna_L,Antenna_R)

if (Antenna_R-Antenna_L>0)
    sign = 1;
elseif (Antenna_R-Antenna_L<0)
    sign = -1;
else
    sign = 0;
end
