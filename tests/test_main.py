#!/usr/bin/env python
# -*- coding: utf-8 -*-
import queue
import threading
from unittest import mock

from click.testing import CliRunner

from hydrus_dd.__main__ import (Consumer, ContentConsumer, HashLoader,
                                ModelLoader, Producer, main)


def test_help():
    runner = CliRunner()
    result = runner.invoke(main, ['--help'])
    assert result.exit_code == 0
    assert result.output


def test_producer():
    hash_ = 'hash'
    with mock.patch('hydrus_dd.__main__.BytesIO'):
        producer = Producer([hash_], mock.Mock())
        producer.start()
        producer.join()
    assert producer.finished.is_set()
    assert not producer.queue.empty()
    item = producer.queue.get()
    assert item[0] == hash_


def test_content_consumer():
    queue_ = queue.Queue()
    hash_ = 'hash'
    queue_.put([hash_, mock.Mock()])
    consumer = ContentConsumer(
        queue_, mock.Mock(), ['tag1'], 0.1, threading.Event())
    consumer.producer_finished.set()
    with mock.patch('hydrus_dd.__main__.evaluate'):
        consumer.start()
        consumer.join()
    assert consumer.finished.is_set()
    assert not consumer.queue.empty()
    item = consumer.queue.get()
    assert item[0] == hash_


def test_consumer():
    queue_ = queue.Queue()
    hash_ = 'hash'
    queue_.put([hash_, ['tag1', 0.1]])
    consumer = Consumer(queue_, mock.Mock(), '{tag}', 'service', mock.Mock(), threading.Event())
    consumer.content_consumer_finished.set()
    consumer.start()
    consumer.join()
    assert queue_.empty()


def test_model_loader():
    model_loader = ModelLoader('model.h5', mock.Mock())
    with mock.patch('hydrus_dd.__main__.tf'):
        model_loader.start()
        model_loader.join()
    assert model_loader.model


def test_hash_loader():
    hash_loader = HashLoader(
        'api_key', 'http://127.0.0.1', ['tag1'], True, True, 10)
    m_client = mock.Mock()
    m_client.search_files.return_value = ['f_id']
    m_client.file_metadata.return_value = [{'hash': 'hash'}]
    with mock.patch('hydrus_dd.__main__.hydrus') as m_hydrus:
        m_hydrus.Client.return_value = m_client
        hash_loader.start()
        hash_loader.join()
    assert hash_loader.hashes
