#include <ctime>
#include <iostream>
class MyTimer
{
public:
    timespec t0, t1;
    MyTimer() {}
    double time_ms;
    double time_s;
    void start()
    {
        clock_gettime(CLOCK_REALTIME, &t0);
    }
    void end()
    {
        clock_gettime(CLOCK_REALTIME, &t1);
        time_ms = t1.tv_sec * 1000 + t1.tv_nsec / 1000000.0 - (t0.tv_sec * 1000 + t0.tv_nsec / 1000000.0);
        time_s = time_ms / 1000;
        std::cout << "   Time: " << time_s << " s" << std::endl;
    }
};