import rsa
import base64


# RSA加密
def rsaEncryption(public_key, cleartext):
    result = rsa.encrypt(cleartext.encode(), public_key)

    return base64.encodebytes(result)


# RSA解密
def rsaDecryption(private_key, ciphertext):
    result = rsa.decrypt(base64.decodebytes(ciphertext), private_key)
    return result.decode()


if __name__ == '__main__':
    # 生成公钥与私钥
    public_key, private_key = rsa.newkeys(1024)

    # 输出公钥、私钥
    print(public_key.save_pkcs1())
    print(private_key.save_pkcs1())

    cleartext = input("请输入明文:")

    # 使用公钥对明文进行加密
    for _ in range(5):
        ciphertext = rsaEncryption(public_key, cleartext)
        print("密文：", ciphertext)

    # 使用私钥对密文进行解密
    decipher = rsaDecryption(private_key, ciphertext)
    print("明文：", decipher)
