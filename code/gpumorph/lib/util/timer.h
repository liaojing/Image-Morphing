#ifndef TIMER_H
#define TIMER_H

#include <string>
#include <list>
#include <cuda_runtime.h>

namespace rod
{

class base_timer/*{{{*/
{
public:
    base_timer(const char *type_label, size_t data_size=0,
               const std::string &unit="");

    void start();
    void stop();
    float elapsed() const;
    bool is_stopped() const { return !m_started; }
    size_t data_size() const { return m_data_size; }
    const std::string &unit() const { return m_unit; }

    const char *type_label() { return m_type_label; }

protected:
    virtual void do_start() = 0;
    virtual void do_stop() = 0;
    virtual float do_get_elapsed() const = 0;

private:
    base_timer(const base_timer &);
    base_timer &operator=(const base_timer &);

    const char *m_type_label;
    mutable float m_elapsed;
    bool m_started;
    size_t m_data_size;
    std::string m_unit;
};/*}}}*/

class gpu_timer : public base_timer/*{{{*/
{
public:
    gpu_timer(size_t data_size=0, const std::string &unit="", bool start=true);
    ~gpu_timer();

private:
    cudaEvent_t m_start, m_stop;

    virtual void do_start();
    virtual void do_stop();
    virtual float do_get_elapsed() const;
};/*}}}*/

class cpu_timer : public base_timer/*{{{*/
{
public:
    cpu_timer(size_t data_size=0, const std::string &unit="", bool start=true);
    ~cpu_timer() {}

private:
    double m_start_time, m_stop_time;

    virtual void do_start();
    virtual void do_stop();
    virtual float do_get_elapsed() const;

    double get_cpu_time() const;
};/*}}}*/

class scoped_timer_stop
{
public:
    scoped_timer_stop(base_timer &timer);
    ~scoped_timer_stop() { stop(); }

    void stop();

    float elapsed() const { return m_timer->elapsed(); }

private:
    base_timer *m_timer;
    static int m_global_padding;
};

class timer_pool
{
public:
    ~timer_pool() { }

    gpu_timer &gpu_add(const std::string &label, size_t data_size=0,
                       const std::string &unit="", bool start=true);
    cpu_timer &cpu_add(const std::string &label, size_t data_size=0,
                       const std::string &unit="", bool start=true);
    void flush();

private:
    struct timer_data
    {
        base_timer *timer;
        std::string label;
        int level;
    };

    typedef std::list<timer_data> timer_list;
    timer_list m_timers;
};

extern timer_pool timers;

} // namespace rod

#endif
