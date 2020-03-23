---
title: Move Your Litecoins Off Coinbase
date: 2018-03-05T11:10:00-08:00
layout: post
tags: ["litecoin", "LTC", "cryptocurrency", "coinbase", "electrum-ltc", "wallets"]
---
Move your Litecoins from Coinbase to a program called Electrum-LTC that runs on your personal computer. This post investigates “why” and “how” to do this.

<!--more-->
To skip to the instructions on how to install and use Electrum-LTC and how to move your Litecoins to it from Coinbase, [click here](#Electrum-LTC-Wallet-on-Mac-OS-X).

## Introduction
Leaving your cryptocurrency on an exchange like [Coinbase](https://www.coinbase.com) is risky. [Many](https://www.quora.com/Is-it-safe-to-leave-Ethereum-coins-in-Coinbase) [have](https://www.reddit.com/r/Bitcoin/comments/79r54g/is_it_safe_to_leave_coins_in_coinbase/) [pointed](https://news.ycombinator.com/item?id=14806636) [out](https://bitcoin.stackexchange.com/questions/10206/if-i-buy-bitcoins-at-coinbase-com-should-i-leave-them-on-coinbase-com-or-shoul/23783) that, as long as your coins are on Coinbase, they are not really yours. 

When you buy coins on Coinbase, you own the coins in a legal sense but you do not control the coins in a technological sense. It's the difference between someone promising to give you $100 versus someone *actually giving you* $100. If someone who has merely promised you the $100 goes bankrupt or flees the country, and all you have is that promise from them, good luck getting your $100.

To actually hold your coins in the technological sense, you need the "private key". When your coins are on Coinbase, they hold the private key and you do not. Without the private key, you only have a promise from Coinbase, not the coins themselves. You don’t want a promise. You want your money.

To break it down for the lawyers out there, you have a legal entitlement to your coins sitting on Coinbase, and you could probably get a court order compelling Coinbase to turn over the private keys. But if Coinbase gets hacked and all of the private keys it holds get stolen, or if Coinbase goes bankrupt or insolvent, you would become a general creditor against Coinbase and would need to get in line with everyone else to get your money back. Or, perhaps worse, if Coinbase simply shut off their website and disappeared one day, you may not be able to find anyone to haul into court. You'll wish you had more than just a legal claim to your coins. You'll wish you had actual possession of your coins -- actual possession meaning the private key that unlocks the wallet containing the coins. 

## Do Not Leave Large Amounts of Coins on Coinbase
The risk of losing your coins if left on Coinbase is not theoretical. This actually happened with one of the first Bitcoin exchanges, [Mt. Gox](https://en.wikipedia.org/wiki/Mt._Gox).

In 2014, Mt. Gox filed for bankruptcy and all of their customers lost access to their Bitcoins stored on the Mt. Gox platform. It was reported that they had lost 850,000 Bitcoins which, it turned out, had been stolen out of Mt. Gox’s wallet over time, beginning in late 2011. If you had a sizable number of Bitcoin stored on Mt. Gox, it might have been worth hiring an attorney and participating as a creditor in the bankruptcy proceedings. Maybe you would have gotten some of your Bitcoins back. Maybe not. You almost certainly would have lost something.

The other (less likely) scenario is what is called an “exit scam.” The exchange can just shut off their website or the I.T. people in charge of the wallets can abscond with the private keys, and you’ve lost your coins. Maybe you can sue Coinbase, Inc. or whatever corporation owes you the money, but by the time you realize that your coins have been stolen, that corporate shell is unlikely to be solvent after losing tens or hundreds of millions of dollars in cryptocurrency to fraud. You can get in line with every other Coinbase customer to recover pennies on the dollar from a bankrupt corporation with few assets at that point.

To avoid this situation, you need to send your coins from Coinbase to a wallet that only you have the private key to. A private key is a just a text file containing hundreds or thousands of random letters and numbers. It’s basically a password, but a really really long one that you could never remember (or type properly) and so needs to be kept in a text file.[^1]

There are, of course, risks to maintaining your own wallet. You could lose the private key. You could get hacked or robbed and someone could take the private keys and send all of your coins to their own wallet. But if you take the precautions outlined in this blog post, the risks of maintaining your own wallet are small compared to the risks of leaving your coins on Coinbase. After all, hackers are constantly trying to hack Coinbase since, if they suceed, the payoff would be enormous. But it is unlikely that hackers are devoting resources to targetting an individual like you where the payoff for stealing your private key would be comparatively small. 

The above rationale for moving your coins off Coinbase to your own wallet applies to any cryptocurrency (Bitcoin, Ethereum, Litecoin, etc), but the discussion of how to move your coins off of Coinbase will focus on Litecoin ("LTC").

## Wallet Options

This section reviews three alternatives to leaving your Litecoin on Coinbase. They are:

* [Electrum-LTC](https://www.electrum-ltc.org), a software wallet that runs on your computer;
* a hardware wallet like the [Ledger Nano S](https://www.amazon.com/Ledger-Nano-Cryptocurrency-Hardware-Wallet/dp/B01J66NF46/); and
* a “paper” wallet that you print that has the private key printed directly on the paper.

The important thing is that you (and only you) have the private key associated with the wallet where your LTC are stored. A software wallet like Electrum-LTC will maintain the private keys on your personal computer.

### Electrum-LTC

Ultimately, I recommend using Electrum-LTC running on your personal computer as the right balance of security, convenience and foolproof-ness.

Without belaboring the details, Electrum-LTC generates a wallet with a corresponding private key and shows you the LTC “address” to send the coins to from Coinbase. Once you’ve sent the coins to this wallet, the Electrum-LTC program running on your computer will quickly detect that you sent the funds and it will display your total balance.

When you quit Electrum-LTC, your wallet and private keys are encrypted so that, even if your computer is hacked or stolen, the thief cannot access the LTC stored on your computer.

Electrum-LTC is open source software and is not owned or controlled by any company and those with appropriate skill can (and have) audited the source code for any malicious code.

### Hardware Wallet

Another option is to purchase a hardware wallet like [Ledger Nano S](https://www.amazon.com/Ledger-Nano-Cryptocurrency-Hardware-Wallet/dp/B01J66NF46/). The price is less than $100, so it’s perfectly affordable if you have large amounts of cryptocurrency. A hardware wallet may also be a good option if you have multiple types of cryptocurrencies (e.g., Bitcoin, Ethereum, Litecoin, etc) and want to store them in the same place.

But there are two main reasons why I do not use hardware wallets. First, for something like the Ledger Nano S, you must install software on your computer anyway to interact with the hardware wallet so it does not offer any convenience advantage over a software wallet like Electrum-LTC.

Second, it is not clear to me how I can confirm that the hardware is trustworthy. How can I be sure that, when I plug the hardware into my computer, it isn’t secretly sending the private keys to my wallets to a third-party server?[^2] As we will see below, with a software wallet like Electrum-LTC, I can verify the checksum and signature of the binary and, as long as I trust the people who signed the binary, I can be reasonably confident that the software isn’t doing anything malicious.

That said, I plan to take a more serious look at hardware wallets in the future. If warranted, I'll update my wallet recommendation.

### Paper Wallet

A paper wallet is, quite literally, the wallet address and private key printed on a piece of paper. You generate the wallet using something like [this site](https://liteaddress.org) and you print it out. It looks something like this:

![Litecoin Paper Wallet](/img/Litecoin-Paper-Wallet.jpg){:style="margin: 0 auto; display: block"}

You could then send your LTC from Coinbase to the public address printed on the paper wallet by typing in the public address or scanning the QR code from the left half of the page. The right half of the page contains your private key and must be kept a secret. Obviously, **don't actually send** any coins to the specific address printed above unless you never want to see your coins again. 

Paper wallets, if used and stored properly, would likely be the most secure (if least convenient) option for storing your LTC. You could store it in a bank safe deposit box or bury it underground or hide it somewhere no one would find it.

If you do choose to go with a paper wallet, there are two very important considerations to be aware of. First, you must be absolutely sure when you generate the paper wallet that your computer or the software you are using to generate the paper wallet has not been compromised. Ideally, you would boot your computer into a fresh, temporary operating system installed on a USB drive, never connect it to the internet, generate and print the paper wallet, and then reboot your computer before reconnecting to the internet. Although this may seem paranoid, please remember that if anyone, anywhere gets ahold of the private key, your LTC are as good as gone.

Second, although it is easy to send funds to the paper wallet, when it comes times to send or spend funds from the paper wallet, you must be careful to “sweep” all of the funds into a software wallet first. If you don’t do this, you may permanently lose some or all of the funds stored in your paper wallet.

I may eventually move my LTC to a paper wallet if I decide I want to hold for the very long term.

## Electrum-LTC Wallet on Mac OS X
As mentioned above, I recommend using Electrum-LTC running on your personal computer to store your LTC. This section details how to install and use Electrum-LTC to get your LTC off of Coinbase. We are focused on how to do this on Mac OS X, but if there is interest in a write-up for Windows or Linux, please send me an email and let me know.

First, go to https://electrum-ltc.org and scroll to the “Downloads” section. Under OS X, click “Executable” to download the .dmg file.

### Security Precautions for the Paranoid

You are about to install a private wallet onto your computer and send your LTC to it. If your computer has been hacked or compromised at the moment you install and setup Electrum-LTC, an attacker could steal your LTC.[^3] But attacker would need to know that you planned to transfer your cryptocurrency to a software wallet and have the skill to compromise your personal computer. Take solace that if you're reading this, you're probably not a high-enough profile target for someone with that kind of information and skill to specifically target you.

A more realistic attack vector is for someone to compromise the Electrum-LTC software you're about to download (rather than your computer itself).[^4] If the program you're about to install has been tampered with, it could be reconfigured to send your private keys to a third party who could then take control of your LTC.

You therefore should confirm that the file you just downloaded is authentic and has not been tampered with. Below are some steps you can take to verify the authenticity. They are not foolproof, and there are ways a dedicated attacker could foil these. I consider it the minimum you need to do to verify the program's authenticity.

1. On the https://electrum-ltc.org website, scroll to the "Downloads" section. Under OS X there is a link that says "checksums". Click on that.
1. There are two long strings of letters and numbers that appear in your browser. An "MD5 hash" and a "SHA-256 hash". Keep this page up on your browser.
1. Open the Terminal app. (You can do this by going to Spotlight and typing `Terminal` and pressing ENTER).
1. Type `cd Downloads` (or wherever the .dmg file was downloaded)
1. Type `md5 electrum-ltc-XXXX.dmg` and press ENTER. Replace XXX with whatever the filename is.
1. Observe that it spits out some random letters and numbers. This is a "hash" of the file. Think of it like a fingerprint. Compare it to the "MD5 hash" that is open in your browser window from the checksum page. You don't need to compare it character-by-character. If the two don't match, they will be very different. 
1. Type `shasum -a 256 electrum-ltc-XXXX.dmg` and press ENTER. Again, replace XXX with whatever the filename is.
1. Observe the second string of long numbers. Another "hash" that just serves as a second check. Compare it to the "SHA-256 hash" open in your browser window from the checksum page. If they do not match it will be very obvious so no need to go character-by-character.

If the MD5 hash and the SHA-256 hash do not match the checksum file open in your browser, it is possible the file you downloaded has been tampered with and you should not go any further.

Now, we've verified that the Electrum-LTC program you've downloaded matches the checksum file. A security-conscious reader may realize that the checksum file itself may have been tampered with, which is why the above steps are not foolproof. If you want to take the extra step and verify that the program you downloaded is certified authentic by the developers, the Electrum-LTC website has [very good instructions](https://electrum-ltc.org/verifying-signatures.html) on how to do that.

### Installing Electrum-LTC

1. Open the `Applications` folder in finder.
1. Open the `Downloads` folder in finder (or wherever the .dmg file was downloaded).
1. Double click the .dmg file to mount it.
1. Drag the Electrum-LTC icon from the mounted .dmg file to the `Applications` folder to install the wallet.
1. Close the .dmg window. Go to your deskop to see the mounted Electrum-LTC drive and drag it to the trash to unmount it.
1. Double click the Electrum-LTC application in your `Applications` folder.
1. It will ask you a bunch of questions like "How do you want to connect to a server?" (Auto Connect), the name of the wallet (default_wallet), the kind of wallet (Standard wallet). You may accept the defaults here.
1. It will ask you whether you want to create a new seed. This is an important step. Assuming this is the first time you're installing Electrum-LTC and your LTC are all on Coinbase, the correct option is "Create a new seed". Click next. The seed type should be "Standard". Click next.

You will now see 12 words written in a box. **Write these words down with a pen on a piece of paper. Do not type them anywhere. Do not store them electronically. _Do not take a picture of them with your phone._** 

This is important. **If you don't do this, and something happens to your computer, your money is GONE.**

These 12 words are not a password. These words allow you to regenerate your wallet with all of your coins in the event your computer crashes or is stolen. This also means that if someone wants to steal your LTC and gets their hands on these 12 words, your money is as good as gone.

In theory you could delete Electrum-LTC from your computer, and as long as you have those 12 words in order, you can always recover your money.

In other words, [keep it secret, keep it safe](https://www.youtube.com/watch?v=_YhpauKGgQ4).

To finish the installation of Electrum-LTC, you will need to type in the 12 words into a confirmation box. Electrum-LTC does this because it knows how important it is that you have a copy of your 12-word seed. 

For the extra cautious, I recommend you write these 12 words down again with a pen on a second piece of paper, and keep these two identical pieces of paper in separate locations. (This will allow you to recover your coins in the event of a fire or flood).

Electrum-LTC will also ask you to set a password. Importantly, this is different from the 12-word seed. This password encrypts the wallet file on your computer so that a thief who gains access to your computer can't just fire up Electrum-LTC while you're not looking and send your coins away to their own wallet. Or if you someday sell your computer but forget to wipe the wallet. As long it's encrypted you're safe. Keep the "Encrypt Wallet" checkbox ticked. (In contrast, the 12-word seed would allow a thief to regenerate your wallet even if they don't have your computer or your encryption password.)


### Sending Money from Coinbase to Electrum-LTC

You open Electrum-LTC like you open any other Mac OS X program. Either from the Applications folder or from Spotlight. From the main screen, you can see your wallet's history and send and receive money. 

1. Click on the "Receive" tab of the Electrum-LTC wallet.
1. You will see a long string of letters and numbers beginning with `L` as the "Receiving address." (**If it doesn't start with `L`, something is wrong. Please stop here.**)
1. Copy this address to your clipboard by highlighting it and pressing Command-C. 
1. Fire up [Coinbase](https://www.coinbase.com).
1. Be sure the URL bar says coinbase.com in it and it is highlighted in green as an encrypted connection.
1. Log into your account.
1. Click on the "Accounts" tab (it's next to "Dashboard" and "Buy/Sell").
1. Under "LTC Wallet" click "Send"
1. In the box that pops up, make sure the "Wallet Address" tab is selected.
1. Under Recipient is says "Enter a BTC Address". In this box, paste the "Receiving address" from Electrum-LTC.
1. Double check that copy/paste happened properly by eyeballing that the first and last few characters of the receiving address copied from Electrum-LTC and pasted into the Coinbase box match. You don't want to have accidentally left off a character from the address or two when you copied.
1. Type the number of LTC you want to send. It doesn't hurt to start with a small amount (e.g., 0.1 LTC) to test the system before you send lots of money.
1. Click "Continue". Take a look at the details and make sure it all checks out.
1. Click "Confirm".
1. After less than 60 seconds, you should see Electrum-LTC report that it received the LTC but that it is "unconfirmed". It will automatically get confirmed once the next LTC block is mined, so once it shows up in your Electrum-LTC application as unconfirmed you should be good to go.

#### Sending Money Out of Electrum-LTC

To get your LTC out of Electrum-LTC, click on the "Send" tab in Electrum-LTC. Here, put in the address  to send to and the number of LTC to send and click "Send". If you've decided to cash out some of your LTC, you can send it to your Coinbase LTC address. Find this by going to your Coinbase account and clicking "Receive" under LTC wallet which will reveal the address to send your LTC to. 

#### Getting a New Computer

If you get a new computer, take care to transfer your wallet over to your new computer before you decommission your old computer. The easy way to transfer your wallet is to simply install Electrum-LTC on your new computer and, during the installation process, don't "create a new seed" but rather regenerate your wallet by selecting "I already have a seed" and inputting the 12-word phrase you carefully wrote down on pen and paper.

You should then delete Electrum-LTC and your wallet off of your old computer so that a future owner of your computer does not have access to your LTC. If you encrypted your wallet as per the instructions above, you're probably safe whether or not you delete Electrum-LTC off you're old computer, but better safe than sorry.

### Parting Thoughts on Electrum-LTC

Electrum-LTC is open source software. If you forget the password to your wallet, there is no one who can help you. There is no "password reset." The only way to recover your LTC in that case would be to regenerate your wallet using your 12-word seed which, if you followed these instructions, has been kept carefully. If you've lost your 12 word seed too, your LTC are gone forever. 

The corollary of this is there are only two ways to steal your LTC. A thief must steal either 
- your 12-word seed OR 
- your wallet password AND your encrypted wallet file that is sitting on your computer. 

If someone just steals your computer that has your wallet file but does not have the wallet password, they will not be able to steal your coins. Don't keep your wallet password on your computer. Keep it in your head. Don't keep your 12-word seed on your computer. Keep it on a piece of paper (or two) in locations different from your computer. 

Be meticulous. You'll thank yourself someday. 

## Footnotes
[^1]: A private key for LTC looks something like: `6u9inNWZkezY7uvd3Db16Kaq8Ck2MuqEum1bKnXpiqYrCJfvw5P`. Don't rely on your memory to store it.
[^2]: There is something called "attestation" that is reported to provide for the verification of the firmware. Frankly, I don't know enough about it to be satisfied with its ability to verify the authenticity of the hardware. 
[^3]: This is because at the moment of installation of Electrum-LTC and its contamporaneous creation of your wallet, your wallet's keys are unusually exposed. 
[^4]: This is a more realistic attack vector because the attacker is targetting all users of Electrum-LTC, who are a much richer target in the aggregate than just you individually. An attacker trying to compromise your specific computer would have to have personal knowledge of your plans. Unles you are very high profile person, there are just not that many of those people in the world who also have the skill and resources to pull off an attack of that sort. 
