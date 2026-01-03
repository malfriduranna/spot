#!/usr/bin/env python3
"""Helper script to download a test video from SoccerNet."""

from SoccerNet.Downloader import SoccerNetDownloader


def main():
    downloader = SoccerNetDownloader(LocalDirectory="data/SoccerNet")
    downloader.password = "ENTER_YOUR_NDA_PASSWORD_HERE"
    downloader.downloadGame(
        files=["1_720p.mkv", "2_720p.mkv"],
        game="england_epl/2016-2017/2016-10-29 - 17-00 West Brom 0 - 4 Manchester City",
    )


if __name__ == "__main__":
    main()
