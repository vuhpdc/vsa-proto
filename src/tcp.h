/**
 * @file tcp.h
 *
 * @date 2020-09-30
 * @author Vinod Nigade (v.v.nigade@vu.nl)
 *
 * @copyright Copyright (c) 2020 VU University, Amsterdam
 *
 */

#ifndef __TCP_H__
#define __TCP_H__

#include <arpa/inet.h>

#include "env_time.h"
#include <memory>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

using namespace std;

class TcpConnection {
public:
  TcpConnection() {}
  TcpConnection(int sock_fd) : sock_fd_m(sock_fd) {}
  virtual int open() {}
  void close() {
    if (sock_fd_m != -1) {
      ::close(sock_fd_m);
    }
    sock_fd_m = -1;
  }

  int send(unsigned char *, size_t) const;
  int recv(unsigned char *, size_t) const;

protected:
  int sock_fd_m = -1;
};

class TcpServer : public TcpConnection {
public:
  TcpServer(unsigned int local_port);
  ~TcpServer();
  int open();
  TcpConnection accept_connection() const;

private:
  unsigned int local_port_m;
};

class TcpClient : public TcpConnection {
public:
  TcpClient(std::string &remote_host, unsigned int remote_port,
            unsigned int local_port);
  ~TcpClient();
  int open();

private:
  std::string remote_host_m;
  unsigned int remote_port_m;
  unsigned int local_port_m;
};

int TcpConnection::recv(unsigned char *read_ptr, size_t size) const {

  unsigned int tot_size = 0;
  while (size > 0) {
    int ret = read(sock_fd_m, read_ptr, size);
    if (ret < 0) {
      if (errno == EINTR || errno == EAGAIN) {
        continue;
      }
      perror("Socket read failure");
      return tot_size;
    } else if (ret == 0) {
      return -1; // remote has closed
    }
    size -= ret;
    read_ptr += ret;
  }

  return tot_size;
}

int TcpConnection::send(unsigned char *send_ptr, size_t size) const {

  while (size > 0) {
    int ret = write(sock_fd_m, send_ptr, size);
    if (ret < 0) {
      if (errno == EINTR || errno == EAGAIN) {
        continue;
      }
      perror("Socket send failure");
      return -1;
    }
    size -= ret;
    send_ptr += ret;
  }
  return 0;
}

/**
 * @brief Construct a new Tcp Server:: Tcp Server object
 *
 * @param remote_host
 * @param remote_port
 * @param local_port
 */
TcpServer::TcpServer(unsigned int local_port) : local_port_m(local_port) {}
TcpServer::~TcpServer() { close(); }

/**
 * @brief open a tcp socket connection
 *
 * @return int
 */
int TcpServer::open() {
  // create socket endpoint
  sock_fd_m = socket(AF_INET, SOCK_STREAM, 0);
  if (sock_fd_m < 0) {
    perror("Socket creation failed");
    return -1;
  }

  int option = 1;
  setsockopt(sock_fd_m, SOL_SOCKET, SO_REUSEADDR, &option, sizeof(option));

  struct sockaddr_in addr;
  addr.sin_family = AF_INET;
  addr.sin_port = htons(local_port_m);
  addr.sin_addr.s_addr = htonl(INADDR_ANY);

  int ret = bind(sock_fd_m, (struct sockaddr *)&addr, sizeof(addr));

  if (ret < 0) {
    goto handle_error;
  }

  ret = listen(sock_fd_m, 5);
  if (ret < 0) {
    goto handle_error;
  }

  return 0;

handle_error:
  perror("Socket open failed");
  return -1;
}

/**
 * @brief Accept client connection.
 *
 * @return int
 */
TcpConnection TcpServer::accept_connection() const {
  struct sockaddr_in client_sock;
  socklen_t sock_len = sizeof(client_sock);

  int client_sock_fd =
      ::accept(sock_fd_m, (struct sockaddr *)&client_sock, &sock_len);

  // disable Nagle's algorithm.
  int option = 1;
  setsockopt(client_sock_fd, IPPROTO_TCP, TCP_NODELAY, &option, sizeof(option));

  if (client_sock_fd < 0) {
    perror("Socket accept connection failed");
    return -1;
  }
  return TcpConnection(client_sock_fd);
}

/**
 * @brief Construct a new Tcp Client:: Tcp Client object
 *
 * @param remote_host
 * @param remote_port
 * @param local_port
 */
TcpClient::TcpClient(std::string &remote_host, unsigned int remote_port,
                     unsigned int local_port)
    : remote_host_m(remote_host), remote_port_m(remote_port),
      local_port_m(local_port) {}

TcpClient::~TcpClient() { close(); }

/**
 * @brief open a tcp socket connection to the remote
 *
 * @return int
 */
int TcpClient::open() {
  // create socket endpoint
  sock_fd_m = socket(AF_INET, SOCK_STREAM, 0);
  if (sock_fd_m < 0) {
    perror("Socket creation failed");
    return -1;
  }

  // disable Nagle's algorithm.
  int option = 1;
  setsockopt(sock_fd_m, IPPROTO_TCP, TCP_NODELAY, &option, sizeof(option));

  struct sockaddr_in addr;
  addr.sin_family = AF_INET;
  addr.sin_port = htons(remote_port_m);
  addr.sin_addr.s_addr = inet_addr(remote_host_m.c_str());
  // connect
  int ret =
      connect(sock_fd_m, (const sockaddr *)&addr, sizeof(struct sockaddr_in));

  if (ret < 0) {
    perror("Socket connect failed");
    return -1;
  }
  return 0;
}

#endif /* __TCP_H__ */