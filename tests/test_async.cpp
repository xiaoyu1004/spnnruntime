#include <iostream>
#include <future>
#include <chrono>

void sayHello(int i)
{
    while (true)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        std::cout << "hello world! thread_id = " << std::this_thread::get_id() << std::endl;
    }
}

static int testAsync()
{
    for (int i = 0; i < 10; ++i)
    {
        std::future<void> ret = std::async(std::launch::async, sayHello, i);
    }
    return 0;
}

int main()
{
    return 0 || testAsync();
}