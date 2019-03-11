function out = timeslice(channel, start, stop)
%TIMESLICE Return a channel slice between the start and stop time

getitem = py.getattr(channel, '__getitem__');
out = getitem(py.slice(start, stop));

end

