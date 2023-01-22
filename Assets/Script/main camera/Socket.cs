using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Threading.Tasks;
using System.Text;

    namespace ServerSocket
    {
        class Program
        {
            static void Main(string[] args)
            {
                // 构建Socket实例、设置端口号和监听队列大小
                var listener = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
                string host = "192.168.5.103";
                int port = 9999;
                listener.Bind(new IPEndPoint(IPAddress.Parse(host), port));
                listener.Listen(5);
                Console.WriteLine("Waiting for connect...");

            // 进入死循环，等待新的客户端连入。一旦有客户端连入，就分配一个Task去做专门处理。然后自己继续等待。
                while(true)
                {
                    var clientExecutor=listener.Accept();
                    Task.Factory.StartNew(()=>
                    {
                        // 获取客户端信息，C#对(ip+端口号)进行了封装。
                        var remote=clientExecutor.RemoteEndPoint;
                        Console.WriteLine("Accept new connection from {0}",remote);

                        // 发送一个欢迎消息
                        clientExecutor.Send(Encoding.UTF8.GetBytes("Welcome"));

                        // 进入死循环，读取客户端发送的信息
                        var bytes=new byte[1024];
                        while(true){
                            var count=clientExecutor.Receive(bytes);
                            var msg=Encoding.UTF8.GetString(bytes,0,count);
                            if(msg=="exit"){
                                System.Console.WriteLine("{0} request close",remote);
                                break;
                            }
                            Console.WriteLine("{0}: {1}",remote,msg);
                            Array.Clear(bytes,0,count);
                        }
                        clientExecutor.Close();
                        System.Console.WriteLine("{0} closed",remote);
                    });
                }
            }
        }
    }


