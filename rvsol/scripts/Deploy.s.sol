// SPDX-License-Identifier: MIT
pragma solidity ^0.8.15;

import { Script } from "forge-std/Script.sol";
import { console2 as console } from "forge-std/console2.sol";

import { Config } from "scripts/Config.sol";
import { Deployer } from "scripts/Deployer.sol";
import { RISCV } from "../src/RISCV.sol";

import { IBigStepper } from "@optimism/src/dispute/interfaces/IBigStepper.sol";
import { IPreimageOracle } from "@optimism/src/cannon/interfaces/IPreimageOracle.sol";
import { DisputeGameFactory } from "@optimism/src/dispute/DisputeGameFactory.sol";
import { DelayedWETH } from "@optimism/src/dispute/weth/DelayedWETH.sol";
import { FaultDisputeGame } from "@optimism/src/dispute/FaultDisputeGame.sol";
import { Safe } from "safe-contracts/Safe.sol";
import { Enum as SafeOps } from "safe-contracts/common/Enum.sol";
import "@optimism/src/libraries/DisputeTypes.sol";

contract Deploy is Deployer {
    /// @notice Modifier that wraps a function in broadcasting.
    modifier broadcast() {
        vm.startBroadcast(msg.sender);
        _;
        vm.stopBroadcast();
    }

    /// @inheritdoc Deployer
    function name() public pure override returns (string memory name_) {
        name_ = "Deploy";
    }

    /// @notice The create2 salt used for deployment of the contract implementations.
    ///         Using this helps to reduce config across networks as the implementation
    ///         addresses will be the same across networks when deployed with create2.
    function _implSalt() internal view returns (bytes32) {
        return keccak256(bytes(Config.implSalt()));
    }

    function run() public {
        deployRiscv();
        setAsteriscFaultGameImplementation(false);
    }

    /// @notice Deploy RISCV
    function deployRiscv() public broadcast returns (address addr_) {
        console.log("Deploying RISCV implementation");
        RISCV riscv = new RISCV{ salt: _implSalt() }(IPreimageOracle(mustGetChainAddress("PreimageOracle")));
        save("RISCV", address(riscv));
        console.log("RISCV deployed at %s", address(riscv));
        addr_ = address(riscv);
    }

    /// @notice Loads the riscv absolute prestate from the prestate-proof for devnets otherwise
    ///         from the config.
    function loadRiscvAbsolutePrestate() internal returns (Claim riscVAbsolutePrestate_) {
        // TODO: Implement this function
        return Claim.wrap(bytes32(0));
    }

    /// @notice Make a call from the Safe contract to an arbitrary address with arbitrary data
    function _callViaSafe(address _target, bytes memory _data) internal {
        Safe safe = Safe(mustGetChainAddress("SystemOwnerSafe"));

        // This is the signature format used the caller is also the signer.
        bytes memory signature = abi.encodePacked(uint256(uint160(msg.sender)), bytes32(0), uint8(1));

        safe.execTransaction({
            to: _target,
            value: 0,
            data: _data,
            operation: SafeOps.Operation.Call,
            safeTxGas: 0,
            baseGas: 0,
            gasPrice: 0,
            gasToken: address(0),
            refundReceiver: payable(address(0)),
            signatures: signature
        });
    }

    /// @notice Sets the implementation for the given fault game type in the `DisputeGameFactory`.
    function setAsteriscFaultGameImplementation(bool _allowUpgrade) public broadcast {
        console.log("Setting Asterisc FaultDisputeGame implementation");
        DisputeGameFactory factory = DisputeGameFactory(mustGetChainAddress("DisputeGameFactoryProxy"));
        DelayedWETH weth = DelayedWETH(mustGetChainAddress("DelayedWETHProxy"));

        if (address(factory.gameImpls(GameTypes.ASTERISC)) != address(0) && !_allowUpgrade) {
            console.log(
                "[WARN] DisputeGameFactoryProxy: `FaultDisputeGame` implementation already set for game type: ASTERISC"
            );
            return;
        }

        FaultDisputeGame fdg = new FaultDisputeGame{ salt: _implSalt() }({
            _gameType: GameTypes.ASTERISC,
            _absolutePrestate: loadRiscvAbsolutePrestate(),
            _genesisBlockNumber: cfg.faultGameGenesisBlock(),
            _genesisOutputRoot: Hash.wrap(cfg.faultGameGenesisOutputRoot()),
            _maxGameDepth :cfg.faultGameMaxDepth(),
            _splitDepth: cfg.faultGameSplitDepth(),
            _gameDuration: Duration.wrap(uint64(cfg.faultGameMaxDuration())),
            _vm: IBigStepper(mustGetAddress("RISCV")),
            _weth: weth,
            _l2ChainId: cfg.l2ChainID()
        });

        bytes memory data = abi.encodeCall(DisputeGameFactory.setImplementation, (GameTypes.ASTERISC, fdg));
        _callViaSafe(address(factory), data);

        console.log(
            "DisputeGameFactoryProxy: set `FaultDisputeGame` implementation (Backend: ASTERISC | GameType: %s)",
            vm.toString(GameType.unwrap(GameTypes.ASTERISC))
        );
    }

}